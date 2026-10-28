"""Health checks, metrics, middleware, security headers, error handling, rate limits."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from bookmind.core.config import Settings
from bookmind.main import create_app
from bookmind.services.registry import ServiceRegistry
from tests.conftest import make_settings

pytestmark = pytest.mark.integration


class TestHealth:
    def test_live(self, client: TestClient) -> None:
        body = client.get("/health/live").json()
        assert body == {"status": "ok", "version": "2.2.0", "env": "test"}

    def test_ready(self, client: TestClient) -> None:
        r = client.get("/health/ready")
        assert r.status_code == 200 and r.json()["status"] == "ready"
        assert r.json()["components"]["semantic"]["ready"] is True

    def test_live_even_when_degraded(self, empty_client: TestClient) -> None:
        assert empty_client.get("/health/live").status_code == 200

    def test_unavailable_when_index_missing(self, empty_client: TestClient) -> None:
        r = empty_client.get("/health/ready")
        assert r.status_code == 503 and r.json()["status"] == "unavailable"

    def test_degraded_when_knn_broken(self, knn_settings: Settings) -> None:
        (knn_settings.artifacts_dir / "model_knn.pkl").write_bytes(b"junk")
        with TestClient(create_app(knn_settings)) as c:
            r = c.get("/health/ready")
        assert r.status_code == 200 and r.json()["status"] == "degraded"


def test_metrics_exposed(client: TestClient) -> None:
    client.get("/api/v1/search?q=dragon")
    text = client.get("/metrics").text
    assert "bookmind_http_requests_total" in text
    assert 'route="/api/v1/search"' in text
    assert "bookmind_http_request_duration_seconds_bucket" in text


class TestRequestContext:
    def test_request_id_generated(self, client: TestClient) -> None:
        rid = client.get("/health/live").headers["X-Request-ID"]
        assert len(rid) == 32

    def test_valid_inbound_request_id_propagated(self, client: TestClient) -> None:
        r = client.get("/api/v1/search?q=", headers={"X-Request-ID": "trace-abc-123"})
        assert r.headers["X-Request-ID"] == "trace-abc-123"
        assert r.json()["error"]["request_id"] == "trace-abc-123"

    @pytest.mark.parametrize("bad", ["short", "x" * 100, "has spaces in it", "<script>xx</script>"])
    def test_malicious_request_id_replaced(self, client: TestClient, bad: str) -> None:
        assert (
            client.get("/health/live", headers={"X-Request-ID": bad}).headers["X-Request-ID"] != bad
        )

    def test_security_headers(self, client: TestClient) -> None:
        h = client.get("/").headers
        assert "script-src 'self'" in h["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in h["Content-Security-Policy"]
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["X-Frame-Options"] == "DENY"
        assert "Strict-Transport-Security" not in h  # only in production

    def test_docs_get_relaxed_csp(self, client: TestClient) -> None:
        assert "cdn.jsdelivr.net" in client.get("/docs").headers["Content-Security-Policy"]

    def test_gzip(self, client: TestClient) -> None:
        r = client.get("/", headers={"Accept-Encoding": "gzip"})
        assert r.headers.get("content-encoding") == "gzip"

    def test_cors(self, client: TestClient) -> None:
        r = client.options(
            "/api/v1/genres",
            headers={"Origin": "http://localhost:8000", "Access-Control-Request-Method": "GET"},
        )
        assert r.headers["access-control-allow-origin"] == "http://localhost:8000"
        r = client.get("/api/v1/genres", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in r.headers


def test_trusted_hosts(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path, trusted_hosts=["bookmind.example.com"]))
    with TestClient(app) as c:
        assert c.get("/health/live").status_code == 400
        assert c.get("/health/live", headers={"host": "bookmind.example.com"}).status_code == 200


class TestProductionMode:
    def test_docs_hidden_and_hsts(self, workdir: Path) -> None:
        prod = make_settings(workdir, env="production")
        with TestClient(create_app(prod)) as c:
            assert c.get("/docs").status_code == 404
            assert c.get("/openapi.json").status_code == 404
            assert "Strict-Transport-Security" in c.get("/health/live").headers

    def test_docs_can_be_enabled_in_production(self, workdir: Path) -> None:
        prod = make_settings(workdir, env="production", enable_docs=True)
        with TestClient(create_app(prod)) as c:
            assert c.get("/docs").status_code == 200
            assert c.get("/openapi.json").status_code == 200


class TestErrorHandling:
    def _app_with_crash(self, settings: Settings) -> TestClient:
        app = create_app(settings, registry=ServiceRegistry.load(settings))
        boom = APIRouter()

        @boom.get("/api/v1/boom")
        def api_boom() -> None:
            raise RuntimeError("secret internal detail")

        @boom.get("/boom")
        def page_boom() -> None:
            raise RuntimeError("secret internal detail")

        app.include_router(boom)
        return TestClient(app, raise_server_exceptions=False)

    def test_500_json_hides_internals(self, settings: Settings) -> None:
        with self._app_with_crash(settings) as c:
            r = c.get("/api/v1/boom")
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "internal_error"
        assert "secret" not in r.text

    def test_500_html(self, settings: Settings) -> None:
        with self._app_with_crash(settings) as c:
            r = c.get("/boom", headers={"accept": "text/html"})
        assert r.status_code == 500 and "<!DOCTYPE html>" in r.text and "secret" not in r.text

    def test_404_json_for_api(self, client: TestClient) -> None:
        r = client.get("/api/v1/nope")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"

    def test_json_when_client_asks(self, client: TestClient) -> None:
        r = client.get("/nope", headers={"accept": "application/json"})
        assert r.json()["error"]["code"] == "not_found"

    def test_preloaded_registry_is_used(self, settings: Settings) -> None:
        reg = ServiceRegistry.load(settings)
        app = create_app(settings, registry=reg)
        with TestClient(app):
            assert app.state.registry is reg


class TestRateLimit:
    def test_limit_enforced_with_envelope(self, workdir: Path) -> None:
        limited = make_settings(workdir, rate_limit_enabled=True, rate_limit="3/minute")
        with TestClient(create_app(limited)) as c:
            codes = [c.get("/api/v1/genres").status_code for _ in range(5)]
            assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
            r = c.get("/api/v1/genres")
            assert r.json()["error"]["code"] == "rate_limited"
            assert 1 <= int(r.headers["Retry-After"]) <= 60
            assert r.headers["X-Request-ID"]  # still traced
            # Browsers get an HTML page, not JSON.
            assert "<!DOCTYPE html>" in c.get("/about", headers={"accept": "text/html"}).text
            # Probes are exempt.
            assert all(c.get("/health/live").status_code == 200 for _ in range(6))
            assert all(c.get("/health/ready").status_code == 200 for _ in range(6))


class TestEmbeddingAndMountedApps:
    """Settings used by the Hugging Face Space: framing by huggingface.co and a mounted /demo."""

    def test_default_forbids_framing(self, workdir: Path) -> None:
        with TestClient(create_app(make_settings(workdir))) as c:
            r = c.get("/health/live")
        assert r.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]

    def test_frame_ancestors_allow_hugging_face(self, workdir: Path) -> None:
        s = make_settings(workdir, frame_ancestors=["https://huggingface.co"])
        with TestClient(create_app(s)) as c:
            r = c.get("/health/live")
        assert "X-Frame-Options" not in r.headers  # CSP frame-ancestors replaces it
        csp = r.headers["Content-Security-Policy"]
        assert "frame-ancestors https://huggingface.co" in csp and "'none'" not in csp
        assert "script-src 'self'" in csp  # the rest of the policy is unchanged

    @pytest.mark.parametrize(
        "bad", [["http://huggingface.co"], ["https://a.com https://b.com"], ["*"]]
    )
    def test_frame_ancestors_validation(self, workdir: Path, bad: list[str]) -> None:
        with pytest.raises(ValueError, match="frame_ancestors"):
            make_settings(workdir, frame_ancestors=bad)

    def test_mounted_app_gets_no_csp_and_no_rate_limit(self, workdir: Path) -> None:
        s = make_settings(workdir, mounted_app_paths=["/demo/"], rate_limit="2/minute")
        assert s.mounted_app_paths == ["/demo"]
        app = create_app(s)
        sub = APIRouter()

        @sub.get("/demo/page")
        def page() -> dict[str, str]:
            return {"ok": "yes"}

        @sub.get("/democracy")
        def other() -> dict[str, str]:
            return {"ok": "no"}

        app.include_router(sub)
        with TestClient(app) as c:
            demo = [c.get("/demo/page") for _ in range(4)]
            other_page = c.get("/democracy")
        assert all(r.status_code == 200 for r in demo)  # not rate limited
        assert "Content-Security-Policy" not in demo[0].headers
        assert demo[0].headers["X-Content-Type-Options"] == "nosniff"  # other headers kept
        # a path that merely starts with the same letters is NOT exempt
        assert "Content-Security-Policy" in other_page.headers

    @pytest.mark.parametrize("bad", [["/"], ["/api"], ["/static/"], ["demo"]])
    def test_mounted_paths_validation(self, workdir: Path, bad: list[str]) -> None:
        with pytest.raises(ValueError, match="mounted_app_paths"):
            make_settings(workdir, mounted_app_paths=bad)
