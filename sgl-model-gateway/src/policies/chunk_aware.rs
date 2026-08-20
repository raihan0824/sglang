/*
    Chunk-Aware Load Balancing Policy

    Places requests by *pending prefill work* rather than by longest prefix match,
    treating prefix affinity as a bounded credit instead of an override.

    Motivation
    ----------
    `cache_aware` scores by longest prefix match. On a workload with one large
    shared prefix, the worker that first computes that prefix wins every
    subsequent decision, and keeps winning: the balance thresholds only intervene
    after the imbalance already exists, and by then the winner also holds the
    cache, so affinity pulls traffic straight back. `round_robin` balances
    perfectly but forfeits the prefill savings of session history.

    chunk_aware scores by the queued prefill backlog, which is known before the
    request is dispatched, and grants prefix affinity only as a small credit that
    a hard spillover bound can override.

    Scoring
    -------
    For each candidate worker w:

        backlog_w  = reported waiting-uncached tokens + router-side reservation
        chunks_w   = backlog_w / chunk_size, scaled by measured prefill rate
        credit_w   = affinity_credit * chunks(match_w), if match rate clears
                     min_cache_match_rate; otherwise 0
        score_w    = load_weight * running_w
                   + prefill_work_weight * chunks_w
                   - credit_w

    and argmin(score_w) wins, round-robin among ties.

    Three rules keep it honest:

    1. Work reservation. The backlog a worker reports is a poll old (1s by
       default). A burst arriving inside one poll window would see identical
       stale backlogs everywhere and stampede onto one worker. So dispatch
       immediately reserves the request's *uncached* token count on the chosen
       worker, released when the response completes. Between polls the router
       reasons about its own dispatches; each poll re-anchors on ground truth.

    2. Bounded spillover. If the backlog gap between the best and worst worker
       exceeds `spillover_bound_chunks`, affinity is ignored entirely and the
       least-backlogged worker wins. This is what makes the cache_aware failure
       mode structurally impossible: affinity can never pin traffic to a worker
       that is already behind.

    3. Long-prefill gate. Affinity applies only to requests at or above
       `long_prefill_threshold_tokens`. Chat-sized traffic routes purely on load
       and backlog, where a cache hit would not have paid for the imbalance.

    Measured prefill rate
    ---------------------
    sglang >= 0.5.16 reports two cumulative counters, `total_prefill_uncached_tokens`
    and `total_prefill_busy_us`, whose delta ratio is a *measured* per-worker
    prefill throughput. That converts a token backlog into a time backlog, which
    is what TTFT actually depends on, and self-calibrates across heterogeneous
    workers without retuning. It is applied as a correction factor relative to the
    fleet median rate, so a homogeneous fleet scores exactly as the plain
    token/chunk_size proxy would, and older workers that do not report the
    counters fall back to that proxy automatically.
*/

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use async_trait::async_trait;
use dashmap::DashMap;
use tracing::{debug, warn};

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    ChunkAwareConfig, LoadBalancingPolicy, PrefillLoadReport, SelectWorkerInfo, WorkerSelection,
};
use crate::core::{Worker, WorkerType};

/// Minimum busy-time delta before a rate sample is trusted, in microseconds.
/// Below this the ratio is dominated by sampling noise and the previous
/// (or fallback) rate is kept instead.
const MIN_RATE_SAMPLE_US: i64 = 1_000;

/// Smoothing factor for the prefill-rate EWMA. Deliberately sluggish: the rate
/// is a hardware property, so a single odd poll should not move it much.
const RATE_EWMA_ALPHA: f64 = 0.3;

fn pool_tag(worker_type: &WorkerType) -> &'static str {
    match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    }
}

/// Trees are keyed `pool::model`, matching `cache_aware`, so prefill/decode/regular
/// pools cannot evict each other's tenants.
fn tree_key_for_worker(worker: &dyn Worker) -> String {
    format!(
        "{}::{}",
        pool_tag(worker.worker_type()),
        normalize_model_key(worker.model_id())
    )
}

/// Rolling per-worker prefill state derived from successive load polls.
#[derive(Debug, Clone, Copy, Default)]
struct PrefillState {
    report: PrefillLoadReport,
    /// Value of the worker's reservation counter at the moment this report was
    /// taken. Reservations at or below this baseline are already reflected in
    /// `report.waiting_uncached_tokens`, so only the excess is added on top —
    /// otherwise a queued request is counted once by the worker and once by the
    /// router for the whole window between the poll and the response finishing.
    reservation_baseline: i64,
    /// Measured prefill throughput in tokens per microsecond, if the worker
    /// reports the cumulative counters and we have two samples to difference.
    rate_tokens_per_us: Option<f64>,
    prev_total_tokens: Option<i64>,
    prev_busy_us: Option<i64>,
}

impl PrefillState {
    /// Fold a fresh report in, updating the measured rate from the delta.
    fn observe(&mut self, report: PrefillLoadReport) {
        let (tokens, busy_us) = match (
            report.total_prefill_uncached_tokens,
            report.total_prefill_busy_us,
        ) {
            (Some(t), Some(b)) => (t, b),
            // Worker predates the counters: keep proxy mode.
            _ => {
                self.report = report;
                return;
            }
        };

        if let (Some(prev_t), Some(prev_b)) = (self.prev_total_tokens, self.prev_busy_us) {
            let d_tokens = tokens - prev_t;
            let d_busy = busy_us - prev_b;
            // Counters reset on worker restart; a negative delta means we are
            // looking at a different process, so drop the stale baseline.
            if d_tokens >= 0 && d_busy >= MIN_RATE_SAMPLE_US {
                let sample = d_tokens as f64 / d_busy as f64;
                if sample.is_finite() && sample > 0.0 {
                    self.rate_tokens_per_us = Some(match self.rate_tokens_per_us {
                        Some(prev) => prev * (1.0 - RATE_EWMA_ALPHA) + sample * RATE_EWMA_ALPHA,
                        None => sample,
                    });
                }
            } else if d_tokens < 0 || d_busy < 0 {
                self.rate_tokens_per_us = None;
            }
        }

        self.prev_total_tokens = Some(tokens);
        self.prev_busy_us = Some(busy_us);
        self.report = report;
    }
}

/// Per-worker inputs to one scoring pass.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    /// Index into the caller's `workers` slice.
    index: usize,
    /// Queued prefill work, expressed in chunk-equivalents.
    backlog_chunks: f64,
    /// Characters of this request already resident on the worker.
    matched_chars: usize,
    /// Uncached tokens this request would add to the worker.
    new_tokens: i64,
    /// Running requests.
    running: f64,
}

#[derive(Debug)]
pub struct ChunkAwarePolicy {
    config: ChunkAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    /// Per-worker-URL prefill state, refreshed by the load monitor.
    prefill: Arc<DashMap<String, PrefillState>>,
    /// Rotates the winner among equally-scored workers.
    tie_breaker: AtomicUsize,
    /// Cached fleet median prefill rate, recomputed on each load update.
    median_rate: RwLock<Option<f64>>,
    _eviction_task: Option<PeriodicTask>,
}

impl ChunkAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(ChunkAwareConfig::default())
    }

    pub fn with_config(config: ChunkAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;
            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "ChunkAwareEviction",
                move || {
                    for tree_ref in trees_clone.iter() {
                        tree_ref.value().evict_tenant_by_size(max_tree_size);
                    }
                },
            ))
        } else {
            None
        };

        Self {
            config,
            trees,
            prefill: Arc::new(DashMap::new()),
            tie_breaker: AtomicUsize::new(0),
            median_rate: RwLock::new(None),
            _eviction_task: eviction_task,
        }
    }

    /// Seed trees for a set of workers (mirrors `CacheAwarePolicy::init_workers`).
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        for worker in workers {
            self.add_worker(worker.as_ref());
        }
    }

    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree = self
            .trees
            .entry(tree_key_for_worker(worker))
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    pub fn remove_worker(&self, worker: &dyn Worker) {
        if let Some(tree) = self.trees.get(&tree_key_for_worker(worker)) {
            tree.remove_tenant(worker.url());
        }
        self.prefill.remove(worker.url());
    }

    pub fn remove_worker_by_url(&self, url: &str) {
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
        self.prefill.remove(url);
    }

    fn chars_to_tokens(&self, chars: usize) -> i64 {
        let per_token = self.config.chars_per_token.max(1.0) as f64;
        (chars as f64 / per_token).round() as i64
    }

    /// Convert a token backlog into chunk-equivalents.
    ///
    /// With a measured rate this is scaled by `median_rate / rate_w`, so a worker
    /// that prefills at half the fleet's rate is charged twice the backlog for the
    /// same token count. On a homogeneous fleet the factor is 1 and this reduces
    /// exactly to `tokens / chunk_size`.
    fn backlog_to_chunks(&self, tokens: i64, rate: Option<f64>, median_rate: Option<f64>) -> f64 {
        let chunk_size = self.config.chunk_size_tokens.max(1) as f64;
        let base = tokens.max(0) as f64 / chunk_size;
        match (rate, median_rate) {
            (Some(r), Some(m)) if r > 0.0 && m > 0.0 => base * (m / r),
            _ => base,
        }
    }

    fn recompute_median_rate(&self) {
        let mut rates: Vec<f64> = self
            .prefill
            .iter()
            .filter_map(|e| e.value().rate_tokens_per_us)
            .filter(|r| *r > 0.0)
            .collect();

        let median = if rates.is_empty() {
            None
        } else {
            rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Some(rates[rates.len() / 2])
        };

        if let Ok(mut guard) = self.median_rate.write() {
            *guard = median;
        }
    }

    /// Gather per-worker scoring inputs for one request.
    fn build_candidates(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        tree: Option<&Arc<Tree>>,
        text: &str,
        est_tokens: i64,
    ) -> Vec<Candidate> {
        let median_rate = self.median_rate.read().ok().and_then(|g| *g);

        healthy_indices
            .iter()
            .map(|&index| {
                let worker = &workers[index];
                let url = worker.url();

                let state = self.prefill.get(url).map(|e| *e.value());
                let reported = state.map(|s| s.report.waiting_uncached_tokens).unwrap_or(0);
                let rate = state.and_then(|s| s.rate_tokens_per_us);

                // The report accounts for everything the worker had queued when
                // it was taken. Add only the reservations made *since* that
                // moment — the dispatches the poll could not have seen.
                let unseen_reservation = (worker.reserved_prefill_tokens()
                    - state.map(|s| s.reservation_baseline).unwrap_or(0))
                .max(0);
                let backlog_tokens = reported + unseen_reservation;

                // `worker.load()` is live and exact for traffic this router
                // dispatched (the load guard maintains it for chunk_aware), while
                // the polled count is up to one interval stale but also sees
                // traffic that bypassed the router. Take the larger: inside a
                // poll window the live counter separates burst arrivals that the
                // stale one would score identically, and the polled value still
                // wins when the worker is busier than we know.
                let running = state
                    .map(|s| s.report.running_reqs)
                    .unwrap_or(0)
                    .max(worker.load() as i64) as f64;

                let matched_chars = match tree {
                    Some(t) if !text.is_empty() => t.prefix_match_tenant_count(text, url),
                    _ => 0,
                };
                let matched_tokens = self.chars_to_tokens(matched_chars);

                Candidate {
                    index,
                    backlog_chunks: self.backlog_to_chunks(backlog_tokens, rate, median_rate),
                    matched_chars,
                    new_tokens: (est_tokens - matched_tokens).max(0),
                    running,
                }
            })
            .collect()
    }

    /// Pick the lowest-scoring candidate, rotating among ties.
    fn argmin_by<F>(&self, candidates: &[Candidate], score: F) -> Option<Candidate>
    where
        F: Fn(&Candidate) -> f64,
    {
        let mut best = f64::INFINITY;
        let mut tied: Vec<Candidate> = Vec::new();

        for candidate in candidates {
            let value = score(candidate);
            if !value.is_finite() {
                continue;
            }
            // Treat near-equal scores as ties so that float noise does not
            // hand one worker a permanent advantage.
            if value < best - f64::EPSILON {
                best = value;
                tied.clear();
                tied.push(*candidate);
            } else if (value - best).abs() <= 1e-9 {
                tied.push(*candidate);
            }
        }

        match tied.len() {
            0 => candidates.first().copied(),
            1 => Some(tied[0]),
            n => {
                let turn = self.tie_breaker.fetch_add(1, Ordering::Relaxed);
                Some(tied[turn % n])
            }
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for ChunkAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.select_worker_detailed(workers, info)
            .await
            .map(|s| s.index)
    }

    async fn select_worker_detailed(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<WorkerSelection> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let text = info.request_text.unwrap_or("");
        let input_chars = text.chars().count();
        let est_tokens = self.chars_to_tokens(input_chars);

        let tree_key = tree_key_for_worker(workers[healthy_indices[0]].as_ref());
        let tree = self.trees.get(&tree_key).map(|e| e.value().clone());
        if tree.is_none() && !text.is_empty() {
            warn!(
                "chunk_aware: no tree for key '{}'; routing on load and backlog only \
                 until the pool tree is seeded",
                tree_key
            );
        }

        let candidates =
            self.build_candidates(workers, &healthy_indices, tree.as_ref(), text, est_tokens);
        if candidates.is_empty() {
            return None;
        }

        // Rule 2: bounded spillover. When the fleet is badly skewed, affinity is
        // off the table entirely — this is the guarantee that chunk_aware cannot
        // reproduce cache_aware's pinning failure.
        let min_backlog = candidates
            .iter()
            .map(|c| c.backlog_chunks)
            .fold(f64::INFINITY, f64::min);
        let max_backlog = candidates
            .iter()
            .map(|c| c.backlog_chunks)
            .fold(f64::NEG_INFINITY, f64::max);
        let spilling = (max_backlog - min_backlog) > self.config.spillover_bound_chunks as f64;

        // Rule 3: long-prefill gate. Short requests never earn affinity.
        let affinity_eligible =
            !spilling && est_tokens >= self.config.long_prefill_threshold_tokens as i64;

        let chunk_size = self.config.chunk_size_tokens.max(1) as f64;
        let min_match_chars =
            (input_chars as f32 * self.config.min_cache_match_rate).ceil() as usize;

        let chosen = if spilling {
            debug!(
                "chunk_aware: spillover (gap {:.2} > {:.2} chunks), ignoring affinity",
                max_backlog - min_backlog,
                self.config.spillover_bound_chunks
            );
            self.argmin_by(&candidates, |c| c.backlog_chunks)?
        } else {
            let load_w = self.config.load_weight as f64;
            let work_w = self.config.prefill_work_weight as f64;
            let credit_w = self.config.prefix_affinity_credit as f64;

            self.argmin_by(&candidates, |c| {
                let credit = if affinity_eligible
                    && c.matched_chars >= min_match_chars
                    && c.matched_chars > 0
                {
                    let matched_chunks = self.chars_to_tokens(c.matched_chars) as f64 / chunk_size;
                    credit_w * matched_chunks
                } else {
                    0.0
                };
                load_w * c.running + work_w * c.backlog_chunks - credit
            })?
        };

        let selected_idx = chosen.index;
        let worker_url = workers[selected_idx].url();

        // Record the request against the chosen worker so subsequent requests can
        // match against it, exactly as cache_aware does.
        if let Some(tree) = tree {
            if !text.is_empty() {
                tree.insert(text, worker_url);
            }
        }

        debug!(
            "chunk_aware: -> {} | est_tokens {} | backlog {:.2} chunks | matched {} chars | \
             reserving {} tokens",
            worker_url, est_tokens, chosen.backlog_chunks, chosen.matched_chars, chosen.new_tokens
        );

        workers[selected_idx].increment_processed();

        Some(WorkerSelection {
            index: selected_idx,
            // Rule 1: reserve only the *uncached* remainder — a matched prefix
            // costs no prefill.
            reserved_prefill_tokens: chosen.new_tokens,
        })
    }

    fn name(&self) -> &'static str {
        "chunk_aware"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_load_updates(&self) -> bool {
        true
    }

    fn load_check_interval_secs(&self) -> Option<u64> {
        Some(self.config.load_check_interval_secs)
    }

    fn update_prefill_loads(
        &self,
        workers: &[Arc<dyn Worker>],
        reports: &HashMap<String, PrefillLoadReport>,
    ) {
        for (url, report) in reports {
            // Snapshot the reservation counter alongside the report so the two
            // describe the same instant.
            let baseline = workers
                .iter()
                .find(|w| w.url() == url)
                .map(|w| w.reserved_prefill_tokens())
                .unwrap_or(0);

            let mut entry = self.prefill.entry(url.clone()).or_default();
            entry.observe(*report);
            entry.reservation_baseline = baseline;
        }
        self.recompute_median_rate();
    }

    fn reset(&self) {
        self.prefill.clear();
        if let Ok(mut guard) = self.median_rate.write() {
            *guard = None;
        }
        self.tie_breaker.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for ChunkAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    const W1: &str = "http://w1:8000";
    const W2: &str = "http://w2:8000";

    fn workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    fn policy_with(config: ChunkAwareConfig) -> ChunkAwarePolicy {
        // Eviction thread off: tests must not race a background task.
        ChunkAwarePolicy::with_config(ChunkAwareConfig {
            eviction_interval_secs: 0,
            ..config
        })
    }

    fn report(waiting_uncached_tokens: i64, running_reqs: i64) -> PrefillLoadReport {
        PrefillLoadReport {
            waiting_uncached_tokens,
            running_reqs,
            total_prefill_uncached_tokens: None,
            total_prefill_busy_us: None,
        }
    }

    /// Deliver a load poll. `workers` is needed so the policy can re-baseline
    /// its reservation accounting against the instant of the report.
    fn set_loads_for(
        policy: &ChunkAwarePolicy,
        workers: &[Arc<dyn Worker>],
        entries: &[(&str, PrefillLoadReport)],
    ) {
        let map: HashMap<String, PrefillLoadReport> = entries
            .iter()
            .map(|(url, r)| ((*url).to_string(), *r))
            .collect();
        policy.update_prefill_loads(workers, &map);
    }

    /// Convenience for tests with no outstanding reservations, where the
    /// baseline is trivially 0.
    fn set_loads(policy: &ChunkAwarePolicy, entries: &[(&str, PrefillLoadReport)]) {
        policy.update_prefill_loads(&[], &map_of(entries));
    }

    fn map_of(entries: &[(&str, PrefillLoadReport)]) -> HashMap<String, PrefillLoadReport> {
        entries
            .iter()
            .map(|(url, r)| ((*url).to_string(), *r))
            .collect()
    }

    fn info<'a>(text: &'a str) -> SelectWorkerInfo<'a> {
        SelectWorkerInfo {
            request_text: Some(text),
            ..Default::default()
        }
    }

    /// A request long enough to clear the default 8192-token gate
    /// (chars_per_token 4.0 => 40k chars ~= 10k tokens).
    fn long_text(seed: &str) -> String {
        seed.repeat(40_000 / seed.len().max(1))
    }

    #[tokio::test]
    async fn scores_on_backlog_when_no_affinity() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        // w1 carries four chunks of queued prefill, w2 is idle.
        set_loads(
            &policy,
            &[(W1, report(4 * 8192, 0)), (W2, report(0, 0))],
        );

        let selected = policy
            .select_worker(&workers, &info("cold request"))
            .await
            .unwrap();
        assert_eq!(selected, 1, "should route to the idle worker");
    }

    #[tokio::test]
    async fn prefix_holder_wins_when_backlog_gap_is_small() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        let text = long_text("shared-prefix-");

        // Seed the prefix onto w1 by routing it there once while w2 is loaded.
        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(8192, 0))]);
        let first = policy.select_worker(&workers, &info(&text)).await.unwrap();
        assert_eq!(first, 0, "first request should land on the idle worker");

        // Now w1 is slightly behind, but by less than the credit is worth:
        // credit = prefix_affinity_credit * match_tokens = 0.2 * ~10000 = ~2000
        // tokens of reach, so a 1000-token deficit should still favour w1.
        set_loads(&policy, &[(W1, report(1000, 0)), (W2, report(0, 0))]);
        let second = policy.select_worker(&workers, &info(&text)).await.unwrap();
        assert_eq!(
            second, 0,
            "prefix holder should win a backlog deficit smaller than the credit's reach"
        );
    }

    /// The affinity credit's reach has a closed form: subtracting
    /// `credit * chunks(match)` from a score measured in `backlog / chunk_size`
    /// means chunk_size cancels, leaving
    ///
    ///     reach_tokens = prefix_affinity_credit * match_tokens
    ///
    /// i.e. at the reference credit of 0.2, a fully-cached 10k-token prompt only
    /// outweighs a ~2000-token backlog gap. This test pins that behaviour so the
    /// tuning sweep has a known baseline rather than a mystery.
    #[tokio::test]
    async fn affinity_credit_reach_is_credit_times_match_tokens() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        let text = long_text("shared-prefix-");
        let est_tokens = policy.chars_to_tokens(text.chars().count());
        let reach = (0.2 * est_tokens as f64) as i64;

        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(8192, 0))]);
        assert_eq!(policy.select_worker(&workers, &info(&text)).await, Some(0));

        // Just inside the reach: affinity holds.
        set_loads(&policy, &[(W1, report(reach - reach / 10, 0)), (W2, report(0, 0))]);
        assert_eq!(
            policy.select_worker(&workers, &info(&text)).await,
            Some(0),
            "deficit just under the reach should keep the prefix holder"
        );

        // Just outside it: the lighter worker takes over.
        set_loads(&policy, &[(W1, report(reach + reach / 2, 0)), (W2, report(0, 0))]);
        assert_eq!(
            policy.select_worker(&workers, &info(&text)).await,
            Some(1),
            "deficit past the reach should spill to the lighter worker"
        );
    }

    #[tokio::test]
    async fn spillover_bound_overrides_affinity() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        let text = long_text("shared-prefix-");

        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(8192, 0))]);
        assert_eq!(policy.select_worker(&workers, &info(&text)).await, Some(0));

        // w1 now holds the prefix *and* a backlog far past the bound (4 chunks).
        set_loads(
            &policy,
            &[(W1, report(20 * 8192, 0)), (W2, report(0, 0))],
        );
        let selected = policy.select_worker(&workers, &info(&text)).await.unwrap();
        assert_eq!(
            selected, 1,
            "spillover must override affinity — this is the cache_aware failure mode"
        );
    }

    #[tokio::test]
    async fn short_requests_ignore_affinity() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        // Short text: well under the 8192-token long-prefill gate.
        let text = "short shared prefix";

        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(4096, 0))]);
        assert_eq!(policy.select_worker(&workers, &info(text)).await, Some(0));

        // w1 holds the prefix but is now (slightly) more loaded. With affinity
        // gated off, the lower backlog must win.
        set_loads(&policy, &[(W1, report(4096, 0)), (W2, report(0, 0))]);
        let selected = policy.select_worker(&workers, &info(text)).await.unwrap();
        assert_eq!(
            selected, 1,
            "short requests must route on load/backlog only"
        );
    }

    #[tokio::test]
    async fn zero_credit_behaves_as_load_only() {
        let policy = policy_with(ChunkAwareConfig {
            prefix_affinity_credit: 0.0,
            ..ChunkAwareConfig::default()
        });
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        let text = long_text("shared-prefix-");

        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(8192, 0))]);
        assert_eq!(policy.select_worker(&workers, &info(&text)).await, Some(0));

        // Same setup as `prefix_holder_wins_when_backlog_gap_is_small`, but with
        // the credit zeroed the prefix holder must lose to the lighter worker.
        set_loads(&policy, &[(W1, report(8192, 0)), (W2, report(0, 0))]);
        let selected = policy.select_worker(&workers, &info(&text)).await.unwrap();
        assert_eq!(selected, 1, "credit 0 => pure load/backlog routing");
    }

    #[tokio::test]
    async fn reserves_uncached_tokens_only() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        let text = long_text("shared-prefix-");
        let est_tokens = policy.chars_to_tokens(text.chars().count());

        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(8192, 0))]);
        let first = policy
            .select_worker_detailed(&workers, &info(&text))
            .await
            .unwrap();
        assert_eq!(first.index, 0);
        assert_eq!(
            first.reserved_prefill_tokens, est_tokens,
            "a cold request reserves its whole token count"
        );

        // Repeat of the same text on the worker that now holds it: the matched
        // prefix costs no prefill, so the reservation must collapse to ~0.
        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(8192, 0))]);
        let second = policy
            .select_worker_detailed(&workers, &info(&text))
            .await
            .unwrap();
        assert_eq!(second.index, 0);
        assert!(
            second.reserved_prefill_tokens < est_tokens / 10,
            "cached repeat should reserve almost nothing, got {}",
            second.reserved_prefill_tokens
        );
    }

    #[tokio::test]
    async fn reservation_shifts_routing_between_polls() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        // Both idle and identical as far as the last poll knows.
        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(0, 0))]);

        // Dispatch a burst without any intervening poll, applying reservations
        // the way WorkerLoadGuard does.
        let mut counts = [0usize; 2];
        for i in 0..8 {
            let text = long_text(&format!("burst-{}-", i));
            let selection = policy
                .select_worker_detailed(&workers, &info(&text))
                .await
                .unwrap();
            counts[selection.index] += 1;
            workers[selection.index]
                .add_reserved_prefill_tokens(selection.reserved_prefill_tokens);
        }

        // Without reservations every request would see backlog 0 everywhere and
        // pile onto one worker.
        assert_eq!(
            counts,
            [4, 4],
            "reservations must spread a burst that arrives inside one poll window"
        );
    }

    #[tokio::test]
    async fn measured_rate_penalizes_the_slower_worker() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        // Two polls are needed before a rate can be differenced.
        let sample = |tokens: i64, busy_us: i64, waiting: i64| PrefillLoadReport {
            waiting_uncached_tokens: waiting,
            running_reqs: 0,
            total_prefill_uncached_tokens: Some(tokens),
            total_prefill_busy_us: Some(busy_us),
        };

        set_loads(
            &policy,
            &[(W1, sample(0, 0, 0)), (W2, sample(0, 0, 0))],
        );
        // w1 prefilled 1M tokens in 1s; w2 managed only 100k in the same second.
        set_loads(
            &policy,
            &[
                (W1, sample(1_000_000, 1_000_000, 2 * 8192)),
                (W2, sample(100_000, 1_000_000, 8192)),
            ],
        );

        // w2 reports the *smaller* token backlog, but is 10x slower, so the
        // time-to-drain favours w1.
        let selected = policy
            .select_worker(&workers, &info("cold request"))
            .await
            .unwrap();
        assert_eq!(
            selected, 0,
            "measured rate should outweigh raw token backlog"
        );
    }

    #[tokio::test]
    async fn rate_baseline_resets_when_counters_go_backwards() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1]);
        policy.init_workers(&workers);

        let sample = |tokens: i64, busy_us: i64| PrefillLoadReport {
            waiting_uncached_tokens: 0,
            running_reqs: 0,
            total_prefill_uncached_tokens: Some(tokens),
            total_prefill_busy_us: Some(busy_us),
        };

        set_loads(&policy, &[(W1, sample(0, 0))]);
        set_loads(&policy, &[(W1, sample(1_000_000, 1_000_000))]);
        assert!(policy.prefill.get(W1).unwrap().rate_tokens_per_us.is_some());

        // Worker restarted: cumulative counters go backwards.
        set_loads(&policy, &[(W1, sample(10, 10))]);
        assert!(
            policy.prefill.get(W1).unwrap().rate_tokens_per_us.is_none(),
            "a counter reset must drop the stale rate rather than infer a wild one"
        );
    }

    /// A request that the worker has acknowledged in its waiting queue must not
    /// also be charged as a router reservation — otherwise the same tokens are
    /// counted twice for the whole window between the poll and the response
    /// completing, and the worker looks far more loaded than it is.
    #[tokio::test]
    async fn poll_rebaselines_reservations_instead_of_double_counting() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        // A request is in flight on w1: 8192 uncached tokens reserved.
        workers[0].add_reserved_prefill_tokens(8192);

        // The poll now observes exactly that request sitting in w1's queue.
        set_loads_for(
            &policy,
            &workers,
            &[(W1, report(8192, 0)), (W2, report(0, 0))],
        );

        // w1's true backlog is 8192, i.e. 1.0 chunk — not 2.0.
        let candidates = policy.build_candidates(
            &workers,
            &[0, 1],
            None,
            "",
            0,
        );
        assert!(
            (candidates[0].backlog_chunks - 1.0).abs() < 1e-9,
            "expected 1.0 chunk after re-baselining, got {}",
            candidates[0].backlog_chunks
        );

        // A *new* dispatch after the poll is unseen by it, so it does add on top.
        workers[0].add_reserved_prefill_tokens(8192);
        let candidates = policy.build_candidates(&workers, &[0, 1], None, "", 0);
        assert!(
            (candidates[0].backlog_chunks - 2.0).abs() < 1e-9,
            "a post-poll dispatch must still count, got {}",
            candidates[0].backlog_chunks
        );
    }

    #[tokio::test]
    async fn skips_unhealthy_workers() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);

        // w2 is idle but unhealthy; w1 is backlogged but up.
        set_loads(&policy, &[(W1, report(100 * 8192, 0)), (W2, report(0, 0))]);
        workers[1].set_healthy(false);

        assert_eq!(policy.select_worker(&workers, &info("x")).await, Some(0));

        workers[0].set_healthy(false);
        assert_eq!(
            policy.select_worker(&workers, &info("x")).await,
            None,
            "no healthy workers => no selection"
        );
    }

    #[tokio::test]
    async fn ties_rotate_between_equal_workers() {
        let policy = policy_with(ChunkAwareConfig::default());
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);
        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(0, 0))]);

        let mut counts = [0usize; 2];
        for _ in 0..10 {
            // Distinct short texts: no affinity, no reservation applied.
            let idx = policy.select_worker(&workers, &info("tie")).await.unwrap();
            counts[idx] += 1;
        }
        assert_eq!(counts, [5, 5], "equal scores must round-robin");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_dispatch_does_not_panic_or_leak_reservations() {
        let policy = Arc::new(policy_with(ChunkAwareConfig::default()));
        let workers = workers(&[W1, W2]);
        policy.init_workers(&workers);
        set_loads(&policy, &[(W1, report(0, 0)), (W2, report(0, 0))]);

        let mut handles = Vec::new();
        for i in 0..64 {
            let policy = Arc::clone(&policy);
            let workers = workers.clone();
            handles.push(tokio::spawn(async move {
                let text = long_text(&format!("c-{}-", i % 8));
                let selection = policy
                    .select_worker_detailed(&workers, &info(&text))
                    .await
                    .unwrap();
                let worker = &workers[selection.index];
                // Reserve then release, as WorkerLoadGuard does.
                worker.add_reserved_prefill_tokens(selection.reserved_prefill_tokens);
                tokio::task::yield_now().await;
                worker.add_reserved_prefill_tokens(-selection.reserved_prefill_tokens);
            }));
        }
        for handle in handles {
            handle.await.expect("dispatch task panicked");
        }

        for worker in &workers {
            assert_eq!(
                worker.reserved_prefill_tokens(),
                0,
                "every reservation must be released"
            );
        }
    }

    #[tokio::test]
    async fn release_cannot_drive_reservation_negative() {
        let workers = workers(&[W1]);
        workers[0].add_reserved_prefill_tokens(100);
        workers[0].add_reserved_prefill_tokens(-500);
        assert_eq!(workers[0].reserved_prefill_tokens(), 0);
    }
}
