//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, StreamExt},
};
use http::StatusCode;
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    core::{metrics_aggregator::MetricPack, ConnectionMode, Worker, WorkerRegistry, WorkerType},
    policies::{PolicyRegistry, PrefillLoadReport},
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/{}", url, endpoint);
            let api_key = worker.api_key().clone();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let url = worker.url().to_string();
                let api_key = worker.api_key().clone();
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill { .. } => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
                let client = client.clone();

                async move {
                    let load = if is_http {
                        Self::parse_load_response(&client, &url, api_key.as_deref()).await
                    } else {
                        -1
                    };
                    WorkerLoadInfo {
                        worker: url,
                        worker_type,
                        load,
                    }
                }
            })
            .collect();

        let loads = future::join_all(futures).await;
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    async fn parse_load_response(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> isize {
        let load_url = format!("{}/v1/loads?include=core", url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(json) => json
                    .get("aggregate")
                    .and_then(|a| a.get("total_tokens"))
                    .and_then(|v| v.as_i64())
                    .map(|n| n as isize)
                    .unwrap_or(-1),
                _ => -1,
            },
            _ => -1,
        }
    }

    /// Fetch the per-worker prefill load snapshot used by chunk_aware.
    ///
    /// Reads the same `/v1/loads?include=core` document as `parse_load_response`,
    /// but keeps the fields that describe *queued prefill work* rather than the
    /// single aggregate token count.
    pub async fn get_all_prefill_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> HashMap<String, PrefillLoadReport> {
        let workers = worker_registry.get_all();

        let futures: Vec<_> = workers
            .iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .map(|worker| {
                let url = worker.url().to_string();
                let api_key = worker.api_key().clone();
                let client = client.clone();

                async move {
                    let report =
                        Self::parse_prefill_load_response(&client, &url, api_key.as_deref()).await;
                    report.map(|r| (url, r))
                }
            })
            .collect();

        future::join_all(futures)
            .await
            .into_iter()
            .flatten()
            .collect()
    }

    async fn parse_prefill_load_response(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> Option<PrefillLoadReport> {
        let load_url = format!("{}/v1/loads?include=core", url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        let response = req.send().await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        let json = response.json::<Value>().await.ok()?;

        // The document holds one entry per DP rank under "loads"; sum across
        // ranks so a DP-sharded worker is comparable to a plain one.
        let entries: Vec<&Value> = match json.get("loads").and_then(|l| l.as_array()) {
            Some(array) => array.iter().collect(),
            None => vec![&json],
        };
        if entries.is_empty() {
            return None;
        }

        let sum = |key: &str| -> Option<i64> {
            let mut total: Option<i64> = None;
            for entry in &entries {
                if let Some(v) = entry.get(key).and_then(|v| v.as_i64()) {
                    total = Some(total.unwrap_or(0) + v);
                }
            }
            total
        };

        Some(PrefillLoadReport {
            waiting_uncached_tokens: sum("num_waiting_uncached_tokens").unwrap_or(0),
            running_reqs: sum("num_running_reqs").unwrap_or(0),
            // Present only on sglang >= 0.5.16; None keeps chunk_aware in
            // fixed-chunk proxy mode instead of measured-rate mode.
            total_prefill_uncached_tokens: sum("total_prefill_uncached_tokens"),
            total_prefill_busy_us: sum("total_prefill_busy_us"),
        })
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match crate::core::metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {}", e)),
        }
    }
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    interval: Duration,
    tx: watch::Sender<HashMap<String, isize>>,
    rx: watch::Receiver<HashMap<String, isize>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let interval = self.interval;
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(worker_registry, policy_registry, client, interval, tx).await;
        });

        *handle_guard = Some(handle);
    }

    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let load_policies = policy_registry.get_all_load_updating_policies();

            if load_policies.is_empty() {
                debug!("No load-aware policies found, skipping load fetch");
                continue;
            }

            // Policies wanting the detailed prefill snapshot (chunk_aware) get a
            // second, richer document; the scalar path stays as-is for
            // power_of_two so its semantics do not shift.
            let wants_prefill_detail = load_policies
                .iter()
                .any(|p| p.name() == "chunk_aware");

            if wants_prefill_detail {
                let reports =
                    WorkerManager::get_all_prefill_loads(&worker_registry, &client).await;
                if reports.is_empty() {
                    warn!("No prefill loads fetched from workers");
                } else {
                    debug!("Fetched prefill loads from {} workers", reports.len());
                    let all_workers = worker_registry.get_all();
                    for policy in &load_policies {
                        policy.update_prefill_loads(&all_workers, &reports);
                    }
                }
            }

            let result = WorkerManager::get_all_worker_loads(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            for load_info in result.loads {
                loads.insert(load_info.worker, load_info.load);
            }

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} load-aware policies",
                    loads.len(),
                    load_policies.len()
                );
                for policy in &load_policies {
                    policy.update_loads(&loads);
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}
