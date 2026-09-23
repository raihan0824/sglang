//! Retry mechanism integration tests
//!
//! Tests for retry behavior: exponential backoff, max retries, and retry-on-failure scenarios.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{RetryConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{clear_fail_status_code, set_fail_status_code},
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
}
