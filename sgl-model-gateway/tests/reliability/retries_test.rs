//! Retry mechanism integration tests
//!
//! Tests for retry behavior: exponential backoff, max retries, and retry-on-failure scenarios.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{PolicyConfig, RetryConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{clear_fail_status_code, fail_next_remaining, set_fail_next, set_fail_status_code},
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

#[cfg(test)]
mod retry_tests {
    use super::*;

    /// Test that retries succeed when at least one worker is healthy
    #[tokio::test]
    async fn test_retry_succeeds_with_healthy_fallback() {
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 100,
            ..Default::default()
        };
        let config = TestRouterConfig::round_robin_with_retry(3300, retry_config);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19200, 1.0), // First worker always fails
                TestWorkerConfig::healthy(19201),    // Second worker always succeeds
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Request should succeed via retry to healthy worker
        let payload = json!({
            "text": "Test retry to healthy worker",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request should succeed via retry"
        );

        ctx.shutdown().await;
    }

    /// Test that retries are disabled when configured
    #[tokio::test]
    async fn test_retries_disabled() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3301)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .disable_retries()
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(19202, 1.0)], // Always fail
        )
        .await;

        let app = ctx.create_app().await;

        // With retries disabled, request should fail immediately
        let payload = json!({
            "text": "Test no retries",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "Request should fail without retries"
        );

        ctx.shutdown().await;
    }

    /// Test max retries limit
    #[tokio::test]
    async fn test_max_retries_limit() {
        let retry_config = RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 10,
            max_backoff_ms: 50,
            ..Default::default()
        };
        let config = TestRouterConfig::round_robin_with_retry(3302, retry_config);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(19203, 1.0)], // Always fail
        )
        .await;

        let app = ctx.create_app().await;

        // All retries will fail, should return error after exhausting retries
        let payload = json!({
            "text": "Test max retries",
            "stream": false
        });

        let start = std::time::Instant::now();
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let elapsed = start.elapsed();

        // Should eventually fail after retries
        assert!(
            resp.status() == StatusCode::INTERNAL_SERVER_ERROR
                || resp.status() == StatusCode::SERVICE_UNAVAILABLE,
            "Should fail after exhausting retries, got {}",
            resp.status()
        );

        // Should take some time due to backoff (at least initial_backoff_ms)
        // With 2 retries and 10ms initial backoff, should take at least 10ms
        // But don't make this too strict as timing can vary
        assert!(
            elapsed.as_millis() >= 5,
            "Should have some backoff delay, got {}ms",
            elapsed.as_millis()
        );

        ctx.shutdown().await;
    }

    /// Test retry with multiple workers - should eventually find healthy one
    #[tokio::test]
    async fn test_retry_finds_healthy_worker() {
        let retry_config = RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 5,
            max_backoff_ms: 50,
            ..Default::default()
        };
        let config = TestRouterConfig::round_robin_with_retry(3303, retry_config);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19204, 1.0), // Fail
                TestWorkerConfig::flaky(19205, 1.0), // Fail
                TestWorkerConfig::healthy(19206),    // Succeed
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // With round robin and retries, should eventually hit the healthy worker
        let payload = json!({
            "text": "Test find healthy worker",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Should succeed by retrying until finding healthy worker"
        );

        ctx.shutdown().await;
    }

    fn cache_aware_with_retry(port: u16, max_retries: u32) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .cache_aware_policy(0.5, 32, 1.5, 60, 1000)
            .host("127.0.0.1")
            .port(port)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .build_unchecked()
    }

    fn generate_request(text: String) -> Request<Body> {
        let payload = json!({ "text": text, "stream": false });
        Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap()
    }

    /// A retry after a 429 goes to another worker. A prefix-affinity policy sends
    /// the same text to the same worker, so without that exclusion the retry of a
    /// request whose worker has a full queue gets the same 429 again.
    #[tokio::test]
    async fn test_retry_after_429_goes_to_another_worker() {
        let full_queue_port = 19650;
        set_fail_status_code(full_queue_port, 429);

        let ctx = AppTestContext::new_with_config(
            cache_aware_with_retry(3304, 2), // two attempts: the first worker, then one other
            vec![
                TestWorkerConfig::flaky(full_queue_port, 1.0), // always 429
                TestWorkerConfig::healthy(19651),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        // Distinct texts, so the policy places some of them on the full worker first.
        for i in 0..10 {
            let text = format!("request {i}: {}", "x".repeat(40 + i));
            let resp = app.clone().oneshot(generate_request(text)).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "request {i} should be retried on the worker that has room"
            );
        }

        clear_fail_status_code(full_queue_port);
        ctx.shutdown().await;
    }

    /// With no other worker left, the retry still goes to the worker that failed
    /// instead of failing with "no available workers".
    #[tokio::test]
    async fn test_retry_exclusion_falls_back_to_the_only_worker() {
        let full_queue_port = 19652;
        set_fail_status_code(full_queue_port, 429);

        let ctx = AppTestContext::new_with_config(
            cache_aware_with_retry(3305, 2),
            vec![TestWorkerConfig::flaky(full_queue_port, 1.0)], // always 429
        )
        .await;
        let app = ctx.create_app().await;

        let resp = app
            .oneshot(generate_request("only worker is full".to_string()))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "the client should get the worker's 429, not a 503"
        );

        clear_fail_status_code(full_queue_port);
        ctx.shutdown().await;
    }

    /// chunk_aware with prefix affinity from the first token, two attempts per
    /// request, and the affinity retry on for requests at least half cached.
    fn chunk_aware_with_affinity_retry(port: u16) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .policy(PolicyConfig::ChunkAware {
                chunk_size_tokens: 8192,
                long_prefill_threshold_tokens: 1,
                load_weight: 1.5,
                prefill_work_weight: 1.0,
                prefix_affinity_credit: 0.2,
                min_cache_match_rate: 0.0,
                spillover_bound_chunks: 4.0,
                load_check_interval_secs: 1,
                chars_per_token: 4.0,
                eviction_interval_secs: 120,
                max_tree_size: 1_000_000,
            })
            .host("127.0.0.1")
            .port(port)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 2,
                initial_backoff_ms: 1,
                max_backoff_ms: 5,
                affinity_min_match: 0.5,
                affinity_backoff_ms: 10,
                ..Default::default()
            })
            .build_unchecked()
    }

    /// A worker that answers 429 to a request whose text it has mostly cached
    /// gets that request again after a short wait (an affinity retry), instead
    /// of the request being served cold by the other worker.
    #[tokio::test]
    async fn test_affinity_retry_returns_to_the_cached_worker() {
        let ports = [19653u16, 19654];
        for port in ports {
            set_fail_status_code(port, 429);
        }

        let ctx = AppTestContext::new_with_config(
            chunk_aware_with_affinity_retry(3306),
            vec![
                TestWorkerConfig::healthy(ports[0]),
                TestWorkerConfig::healthy(ports[1]),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        // First turn: lands on one worker, which now holds the text in the tree.
        let text = format!("conversation so far: {}", "y".repeat(200));
        let resp = app
            .clone()
            .oneshot(generate_request(text.clone()))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Both workers refuse their next request. Without the affinity rule the
        // retry goes to the other worker and meets its refusal (two attempts,
        // both 429); with it, the retry waits and returns to the first worker,
        // which is free again, and the other worker is never tried.
        for port in ports {
            set_fail_next(port, 1);
        }
        let resp = app.clone().oneshot(generate_request(text)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "the affinity retry should be served by the worker holding the cache"
        );
        let untouched: usize = ports.iter().map(|&port| fail_next_remaining(port)).sum();
        assert_eq!(untouched, 1, "the other worker must not have been tried");

        // A cold request (nothing cached anywhere) still moves to the other
        // worker after a refusal, as before.
        for port in ports {
            set_fail_next(port, 0);
        }
        let cold = format!("brand new conversation: {}", "z".repeat(200));
        let resp = app
            .clone()
            .oneshot(generate_request(cold.clone()))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        for port in ports {
            set_fail_next(port, 0);
            clear_fail_status_code(port);
        }
        ctx.shutdown().await;
    }

    /// A refused attempt must not count as "seen" on the worker that refused
    /// it. One worker always answers 429: a cold request refused there is
    /// served by the other worker, and the client's retry of the same text must
    /// go straight to the worker that served it. If the refusing worker kept
    /// the record, the retry would look fully cached there, be pinned to it by
    /// the affinity rule, and fail with the same 429 about half the time.
    #[tokio::test]
    async fn test_refused_attempt_is_not_remembered_as_seen() {
        let full_port = 19655;
        set_fail_status_code(full_port, 429);

        let ctx = AppTestContext::new_with_config(
            chunk_aware_with_affinity_retry(3307),
            vec![
                TestWorkerConfig::flaky(full_port, 1.0), // always 429
                TestWorkerConfig::healthy(19656),
            ],
        )
        .await;
        let app = ctx.create_app().await;

        for i in 0..12 {
            let text = format!("conversation {i}: {}", "q".repeat(120 + i));
            for turn in 0..2 {
                let resp = app
                    .clone()
                    .oneshot(generate_request(text.clone()))
                    .await
                    .unwrap();
                assert_eq!(
                    resp.status(),
                    StatusCode::OK,
                    "conversation {i} send {turn} should be served by the worker with room"
                );
            }
        }

        clear_fail_status_code(full_port);
        ctx.shutdown().await;
    }
}
