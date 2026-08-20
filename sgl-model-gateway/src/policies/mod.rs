//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use std::{fmt::Debug, sync::Arc};

use async_trait::async_trait;
use smg_mesh::OptionalMeshSyncManager;

use crate::core::{HashRing, Worker};

mod bucket;
mod cache_aware;
mod chunk_aware;
mod consistent_hashing;
mod factory;
mod manual;
mod power_of_two;
mod prefix_hash;
mod random;
mod registry;
mod round_robin;
pub mod tree;
pub(crate) mod utils;
pub use bucket::BucketPolicy;
pub use cache_aware::CacheAwarePolicy;
pub use chunk_aware::ChunkAwarePolicy;
pub use consistent_hashing::ConsistentHashingPolicy;
pub use factory::PolicyFactory;
pub use manual::{ManualConfig, ManualPolicy};
pub use power_of_two::PowerOfTwoPolicy;
pub use prefix_hash::{PrefixHashConfig, PrefixHashPolicy};
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use round_robin::RoundRobinPolicy;
pub use tree::PrefixMatchResult;

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
#[async_trait]
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    /// Now uses Arc<dyn Worker> for better performance and to avoid unnecessary cloning.
    ///
    /// # Arguments
    /// * `workers` - Available workers to select from
    /// * `info` - Additional information for routing decisions
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize>;

    /// Select a worker and report any side-effects the caller must apply.
    ///
    /// Policies that reserve work at dispatch time (chunk_aware) need to tell the
    /// router how much, so the router can tie the release to the response
    /// lifetime via `WorkerLoadGuard`. The default delegates to `select_worker`
    /// and reserves nothing, so existing policies are unaffected.
    async fn select_worker_detailed(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<WorkerSelection> {
        self.select_worker(workers, info)
            .await
            .map(WorkerSelection::at)
    }

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// Whether this policy needs the `LoadMonitor` to poll worker load.
    ///
    /// The monitor skips the whole fetch when no registered policy wants it.
    fn needs_load_updates(&self) -> bool {
        false
    }

    /// Preferred load poll interval in seconds, if this policy has an opinion.
    fn load_check_interval_secs(&self) -> Option<u64> {
        None
    }

    /// Receive the detailed per-worker prefill load snapshot.
    ///
    /// Richer counterpart to `update_loads`, which can only carry a single
    /// scalar per worker. `workers` is passed alongside the reports so a policy
    /// can re-baseline its own router-side accounting against the moment the
    /// snapshot was taken. Default is a no-op.
    fn update_prefill_loads(
        &self,
        _workers: &[Arc<dyn Worker>],
        _reports: &std::collections::HashMap<String, PrefillLoadReport>,
    ) {
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Check if this policy needs request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &std::collections::HashMap<String, isize>) {
        // Default: no-op for policies that don't use load information
    }

    /// Set mesh sync manager
    fn set_mesh_sync(&mut self, _mesh_sync: OptionalMeshSyncManager) {
        // Default: no-op for policies that don't use mesh sync
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// A worker choice plus the side-effects the router must apply for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerSelection {
    /// Index into the `workers` slice passed to the policy.
    pub index: usize,
    /// Uncached prefill tokens to reserve on the chosen worker until the
    /// response completes. 0 for policies that do not reserve.
    pub reserved_prefill_tokens: i64,
}

impl WorkerSelection {
    /// A selection with no reservation.
    pub fn at(index: usize) -> Self {
        Self {
            index,
            reserved_prefill_tokens: 0,
        }
    }
}

/// Per-worker prefill load, as reported by the worker's `/v1/loads?include=core`.
///
/// Field names mirror sglang's `LoadSnapshot`. The two cumulative counters exist
/// only on sglang >= 0.5.16; they are `None` against older workers, in which case
/// chunk_aware falls back to a fixed chunk-size proxy instead of a measured rate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PrefillLoadReport {
    /// Uncached input tokens queued for prefill compute (`num_waiting_uncached_tokens`).
    pub waiting_uncached_tokens: i64,
    /// Requests currently running (`num_running_reqs`).
    pub running_reqs: i64,
    /// Cumulative tokens prefilled (`total_prefill_uncached_tokens`).
    pub total_prefill_uncached_tokens: Option<i64>,
    /// Cumulative microseconds spent prefilling (`total_prefill_busy_us`).
    pub total_prefill_busy_us: Option<i64>,
}

/// Configuration for the chunk-aware policy.
///
/// Defaults match the reference deployment's flags.
#[derive(Debug, Clone)]
pub struct ChunkAwareConfig {
    /// Prefill chunk size, in tokens. Used as the unit that converts a token
    /// backlog into a comparable "chunks of queued work" score.
    pub chunk_size_tokens: usize,
    /// Requests below this token count route on load + backlog only, ignoring
    /// prefix affinity entirely.
    pub long_prefill_threshold_tokens: usize,
    /// Weight on running-request count.
    pub load_weight: f32,
    /// Weight on queued prefill work.
    pub prefill_work_weight: f32,
    /// Weight on the prefix-affinity credit (subtracted from the score).
    pub prefix_affinity_credit: f32,
    /// Minimum prefix match rate before any affinity credit is granted.
    pub min_cache_match_rate: f32,
    /// If the backlog gap between best and worst worker exceeds this many
    /// chunks, ignore affinity and route to the least-backlogged worker.
    pub spillover_bound_chunks: f32,
    /// Worker load poll interval.
    pub load_check_interval_secs: u64,
    /// Characters per token, used to estimate token counts from request text.
    /// The router's radix tree stores characters, not token ids (see
    /// `cache_aware`), so token counts are estimated rather than exact.
    pub chars_per_token: f32,
    /// Radix tree eviction interval; shared semantics with cache_aware.
    pub eviction_interval_secs: u64,
    /// Max radix tree size; shared semantics with cache_aware.
    pub max_tree_size: usize,
}

impl Default for ChunkAwareConfig {
    fn default() -> Self {
        Self {
            chunk_size_tokens: 8192,
            long_prefill_threshold_tokens: 8192,
            load_weight: 1.5,
            prefill_work_weight: 1.0,
            prefix_affinity_credit: 0.2,
            min_cache_match_rate: 0.3,
            spillover_bound_chunks: 4.0,
            load_check_interval_secs: 1,
            chars_per_token: 4.0,
            eviction_interval_secs: 120,
            max_tree_size: 67_108_864,
        }
    }
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub bucket_adjust_interval_secs: usize,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 5,
        }
    }
}

/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
        .map(|(idx, _)| idx)
        .collect()
}

/// Helper function to normalize model_id to a key for policy lookups.
///
/// Returns UNKNOWN_MODEL_ID for empty model_ids to ensure consistent behavior
/// across single-model and multi-model deployments.
#[inline]
pub(crate) fn normalize_model_key(model_id: &str) -> &str {
    if model_id.is_empty() {
        crate::core::UNKNOWN_MODEL_ID
    } else {
        model_id
    }
}

/// Information passed to policy for worker selection
#[derive(Debug, Clone, Default)]
pub struct SelectWorkerInfo<'a> {
    /// Request text for cache-aware routing
    pub request_text: Option<&'a str>,
    /// Tokenized request for prefix-hash routing
    /// Used by PrefixHashPolicy for token-based prefix hashing
    pub tokens: Option<&'a [u32]>,
    /// HTTP headers for header-based routing policies
    /// Policies can extract routing information from headers like:
    /// - X-SMG-Target-Worker: Direct routing to a specific worker by index
    /// - X-SMG-Routing-Key: Consistent hash routing for session affinity
    pub headers: Option<&'a http::HeaderMap>,
    /// Pre-computed hash ring for O(log n) consistent hashing
    /// Built and cached by WorkerRegistry, passed through to avoid per-request rebuilds
    pub hash_ring: Option<Arc<HashRing>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key2")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_healthy(false);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }
}
