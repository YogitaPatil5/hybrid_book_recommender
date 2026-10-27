"""Tests for scripts/prepare_hf_space.py (Hugging Face Spaces packaging)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prepare_hf_space", ROOT / "scripts" / "prepare_hf_space.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hf = _load_script()


FAKE_VERSIONS = {
    "numpy": ("2.1.0", ">=3.10"),
    "pandas": ("2.2.3", ">=3.9"),
    "scikit-learn": ("1.6.1", ">=3.9"),
    "scipy": ("1.15.0", ">=3.10"),
    "faiss-cpu": ("1.9.0", ">=3.9"),
    "pyarrow": ("19.0.0", ">=3.9"),
    "sentence-transformers": ("4.1.0", ">=3.9"),
    "pydantic": ("2.10.0", ">=3.8"),
    "pydantic-settings": ("2.7.0", ">=3.8"),
}


@pytest.fixture(autouse=True)
def _fake_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic 'locally installed' versions, and the same results on a PC and in CI:
    GitHub Actions sets GITHUB_ACTIONS=true, which adds a CI-only warning to prepare()."""
    monkeypatch.setattr(hf, "_installed", FAKE_VERSIONS.get)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


def _fake_project(tmp: Path, *, backend: str = "sentence-transformers", knn: bool = True) -> Path:
    """A minimal copy of the project layout with fake artifacts."""
    root = tmp / "project"
    for name in hf.PROJECT_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, root / name)
    shutil.copytree(ROOT / "deploy", root / "deploy")
    pkg = root / "src" / "bookmind"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n")
    for sub in (*hf.GRADIO_PACKAGES, "api", "web", "pipeline"):
        (pkg / sub).mkdir()
        (pkg / sub / "__init__.py").write_text("")
    (pkg / "services" / "__pycache__").mkdir()
    (pkg / "services" / "__pycache__" / "junk.pyc").write_bytes(b"0")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "junk.pyc").write_bytes(b"0")
    art = root / "artifacts"
    art.mkdir()
    (art / "book_index.faiss").write_bytes(b"i" * 10)
    (art / "book_meta.parquet").write_bytes(b"m" * 10)
    (art / "manifest.json").write_text(json.dumps({"embedding_backend": backend}))
    if knn:
        for name in hf.KNN_ARTIFACTS:
            (art / name).write_bytes(b"k")
    (art / "knn_eval_binary.json").write_text("{}")  # not needed in the Space
    return root


class TestPrepareGradio:
    def test_builds_flat_gradio_space(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        out = tmp_path / "space"
        assert hf.prepare(root, out) == []  # gradio is the default

        names = {p.name for p in out.iterdir()}
        expected = {"app.py", "README.md", "requirements.txt", "bookmind.zip"}
        assert names == expected | {*hf.REQUIRED_ARTIFACTS, *hf.KNN_ARTIFACTS}
        assert not [p for p in out.iterdir() if p.is_dir()]  # no folders to upload
        card = (out / "README.md").read_text(encoding="utf-8")
        assert "sdk: gradio" in card and "app_file: app.py" in card
        assert f"sdk_version: {hf.GRADIO_SDK_VERSION}" in card
        assert f'python_version: "{hf.SPACE_PYTHON}"' in card
        assert "{" not in card.split("---")[1]  # every placeholder filled
        assert "gr.Blocks" in (out / "app.py").read_text(encoding="utf-8")

    def test_zip_contains_only_service_layer_python(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        out = tmp_path / "space"
        hf.prepare(root, out)
        with zipfile.ZipFile(out / "bookmind.zip") as zf:
            names = set(zf.namelist())
        assert "bookmind/__init__.py" in names
        assert {n.split("/")[1] for n in names if n.count("/") == 2} == set(hf.GRADIO_PACKAGES)
        assert all(n.endswith(".py") for n in names)
        assert not any(part in n for n in names for part in ("api/", "web/", "pipeline/"))

    def test_zip_is_importable(self, tmp_path: Path) -> None:
        out = tmp_path / "space"
        hf._zip_package(ROOT / "src" / "bookmind", tmp_path / "bookmind.zip")
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import bookmind, bookmind.services.validation as v; "
            "print(bookmind.__file__); print(v.clean_text('  hi  '))"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(tmp_path / "bookmind.zip")],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "bookmind.zip" in result.stdout and "hi" in result.stdout
        assert not out.exists()

    def test_requirements_pin_local_versions(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        out = tmp_path / "space"
        hf.prepare(root, out)
        reqs = (out / "requirements.txt").read_text(encoding="utf-8")
        for package, (version, _) in FAKE_VERSIONS.items():
            assert f"{package}=={version}" in reqs
        assert "gradio" not in reqs.split("\n", 3)[-1]  # comes from sdk_version
        assert "fastapi" not in reqs


class TestRequirements:
    def test_missing_library_uses_fallback_and_warns_for_pickle_libs(self) -> None:
        versions = dict(FAKE_VERSIONS)
        del versions["pandas"], versions["pyarrow"]
        text, warnings = hf.gradio_requirements(lookup=versions.get)
        assert hf.FALLBACK_REQUIREMENTS["pandas"] in text
        assert hf.FALLBACK_REQUIREMENTS["pyarrow"] in text
        assert len(warnings) == 1 and "pandas" in warnings[0]

    def test_version_without_wheels_for_space_python_is_not_pinned(self) -> None:
        versions = dict(FAKE_VERSIONS)
        versions["numpy"] = ("9.0.0", ">=3.14")
        text, warnings = hf.gradio_requirements(lookup=versions.get, python="3.12")
        assert "numpy==9.0.0" not in text
        assert hf.FALLBACK_REQUIREMENTS["numpy"] in text
        assert "numpy 9.0.0" in warnings[0] and ">=3.14" in warnings[0]

    def test_no_requires_python_means_supported(self) -> None:
        assert hf._supports_python(None, "3.12")
        assert hf._supports_python(">=3.10", "3.12")
        assert not hf._supports_python(">=3.13", "3.12")

    def test_real_lookup_reads_installed_metadata(self) -> None:
        module = _load_script()  # fresh, unpatched copy
        found = module._installed("pytest")
        assert found is not None and found[0].count(".") >= 1
        assert module._installed("definitely-not-a-real-package-xyz") is None


class TestPrepareDocker:
    def test_builds_complete_docker_space(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        out = tmp_path / "space"
        assert hf.prepare(root, out, sdk="docker") == []

        for name in (*hf.PROJECT_FILES, *hf.SPACE_FILES):
            assert (out / name).is_file()
        assert "sdk: docker" in (out / "README.md").read_text(encoding="utf-8")
        assert "7860" in (out / "Dockerfile").read_text(encoding="utf-8")
        assert (out / "src" / "bookmind" / "__init__.py").is_file()
        assert not list((out / "src").rglob("__pycache__"))
        copied = {p.name for p in (out / "artifacts").iterdir()}
        assert copied == {*hf.REQUIRED_ARTIFACTS, *hf.KNN_ARTIFACTS}

    def test_unknown_sdk(self, tmp_path: Path) -> None:
        with pytest.raises(hf.PrepareError, match="streamlit"):
            hf.prepare(_fake_project(tmp_path), tmp_path / "space", sdk="streamlit")


class TestPrepareCommon:
    def test_rerun_replaces_old_folder(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        out = tmp_path / "space"
        out.mkdir()
        (out / "stale.txt").write_text("old")
        hf.prepare(root, out)
        assert not (out / "stale.txt").exists()

    def test_missing_knn_is_a_warning_not_an_error(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path, knn=False)
        warnings = hf.prepare(root, tmp_path / "space")
        assert len(warnings) == 1 and "k-NN" in warnings[0]
        assert not (tmp_path / "space" / "model_knn.pkl").exists()

    def test_missing_index_is_an_error(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        (root / "artifacts" / "book_index.faiss").unlink()
        with pytest.raises(hf.PrepareError, match=r"book_index\.faiss"):
            hf.prepare(root, tmp_path / "space")

    def test_hashing_index_is_rejected(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path, backend="hashing")
        with pytest.raises(hf.PrepareError, match="hashing"):
            hf.prepare(root, tmp_path / "space")

    @pytest.mark.parametrize("target", [".", "..", "src", "artifacts", "src/bookmind"])
    def test_refuses_dangerous_output_folders(self, tmp_path: Path, target: str) -> None:
        root = _fake_project(tmp_path)
        with pytest.raises(hf.PrepareError):
            hf.prepare(root, root / target)
        assert (root / "src" / "bookmind" / "__init__.py").exists()

    def test_allows_new_folder_inside_project(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        hf.prepare(root, root / "hf_space")
        assert (root / "hf_space" / "app.py").exists()


class TestMain:
    def test_success_prints_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _fake_project(tmp_path)
        code = hf.main(["--root", str(root), "--out", str(tmp_path / "space")])
        out = capsys.readouterr().out
        assert code == 0
        assert "Ready (gradio Space)" in out and "book_index.faiss" in out
        assert "bookmind.zip" in out and "ALL files listed above" in out
        assert "Upload files" in out and "DEPLOY_HUGGINGFACE.md" in out

    def test_docker_flag(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = _fake_project(tmp_path)
        argv = ["--root", str(root), "--out", str(tmp_path / "s"), "--sdk", "docker"]
        assert hf.main(argv) == 0
        assert "Ready (docker Space)" in capsys.readouterr().out
        assert (tmp_path / "s" / "Dockerfile").exists()

    def test_error_returns_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = _fake_project(tmp_path)
        (root / "artifacts" / "manifest.json").unlink()
        code = hf.main(["--root", str(root), "--out", str(tmp_path / "space")])
        assert code == 1
        assert "pipeline.cli all" in capsys.readouterr().err

    def test_warning_and_lfs_note(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = _fake_project(tmp_path, knn=False)
        (root / "artifacts" / "book_index.faiss").write_bytes(b"0" * (11 * 1_048_576))
        assert hf.main(["--root", str(root), "--out", str(tmp_path / "space")]) == 0
        out = capsys.readouterr().out
        assert "Git LFS" in out and "WARNING" in out


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"{status_code} Client Error")
        self.response = FakeResponse(status_code)


class LocalTokenNotFoundError(Exception):
    """Same class name huggingface_hub raises when no token is saved."""


class FakeApi:
    def __init__(
        self,
        logged_in: bool = True,
        fail_upload: bool = False,
        flaky: int = 0,
        status: int = 401,
    ) -> None:
        self.logged_in, self.fail_upload, self.flaky, self.status = (
            logged_in,
            fail_upload,
            flaky,
            status,
        )
        self.calls: list[tuple[str, dict[str, object]]] = []

    def whoami(self) -> dict[str, str]:
        if self.flaky:
            self.flaky -= 1
            raise ConnectionResetError("[WinError 10054] connection forcibly closed")
        if not self.logged_in:
            raise HTTPError(self.status)
        return {"name": "tester"}

    def create_repo(self, repo_id: str, **kw: object) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kw}))

    def upload_folder(self, **kw: object) -> None:
        if self.fail_upload:
            raise ConnectionError("network down")
        self.calls.append(("upload_folder", kw))


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf.time, "sleep", lambda _s: None)


class TestUpload:
    def test_creates_docker_space_and_uploads(self, tmp_path: Path) -> None:
        api = FakeApi()
        url = hf.upload(tmp_path, "tester/bookmind", api=api)
        assert url == "https://huggingface.co/spaces/tester/bookmind"
        (name1, create), (name2, up) = api.calls
        assert name1 == "create_repo" and create["space_sdk"] == "gradio"
        assert create["repo_type"] == "space" and create["exist_ok"] is True
        assert name2 == "upload_folder" and up["folder_path"] == str(tmp_path)

    def test_docker_sdk_is_passed_through(self, tmp_path: Path) -> None:
        api = FakeApi()
        hf.upload(tmp_path, "tester/bookmind", api=api, sdk="docker")
        assert api.calls[0][1]["space_sdk"] == "docker"

    def test_last_attempt_client_error_is_not_called_network(self, tmp_path: Path) -> None:
        class FlakyThenBad(FakeApi):
            attempts = 0

            def create_repo(self, repo_id: str, **kw: object) -> None:
                self.attempts += 1
                if self.attempts < 3:
                    raise ConnectionResetError("reset")
                raise HTTPError(402)

        with pytest.raises(hf.PrepareError) as info:
            hf.upload(tmp_path, "tester/bookmind", api=FlakyThenBad())
        assert "402" in str(info.value) and "network problem" not in str(info.value)

    @pytest.mark.parametrize("repo", ["bookmind", "a/b/c", "/bookmind", "user/"])
    def test_rejects_bad_repo_id(self, tmp_path: Path, repo: str) -> None:
        with pytest.raises(hf.PrepareError, match="username"):
            hf.upload(tmp_path, repo, api=FakeApi())

    @pytest.mark.parametrize("status", [401, 403])
    def test_bad_or_read_only_token(self, tmp_path: Path, status: int) -> None:
        with pytest.raises(hf.PrepareError, match="WRITE token"):
            hf.upload(tmp_path, "tester/bookmind", api=FakeApi(logged_in=False, status=status))

    def test_no_saved_token(self, tmp_path: Path) -> None:
        class NoToken(FakeApi):
            def whoami(self) -> dict[str, str]:
                raise LocalTokenNotFoundError("no token")

        with pytest.raises(hf.PrepareError, match="WRITE token"):
            hf.upload(tmp_path, "tester/bookmind", api=NoToken())

    def test_network_drop_is_retried(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api = FakeApi(flaky=2)  # fails twice, then works
        assert hf.upload(tmp_path, "tester/bookmind", api=api).endswith("tester/bookmind")
        out = capsys.readouterr().out
        assert out.count("retrying") == 2 and "WinError 10054" in out
        assert [c[0] for c in api.calls] == ["create_repo", "upload_folder"]

    def test_persistent_network_error_is_not_called_a_login_problem(self, tmp_path: Path) -> None:
        with pytest.raises(hf.PrepareError) as info:
            hf.upload(tmp_path, "tester/bookmind", api=FakeApi(flaky=5))
        msg = str(info.value)
        assert "network problem" in msg and "WinError 10054" in msg
        assert "WRITE token" not in msg

    def test_client_error_shows_server_message_without_retry(self, tmp_path: Path) -> None:
        class BadName(FakeApi):
            attempts = 0

            def create_repo(self, repo_id: str, **kw: object) -> None:
                self.attempts += 1
                raise HTTPError(400)

        api = BadName()
        with pytest.raises(hf.PrepareError, match="400 Client Error") as info:
            hf.upload(tmp_path, "tester/bookmind", api=api)
        assert "create the Space" in str(info.value)
        assert api.attempts == 1

    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_rate_limit_and_server_errors_are_retried(self, tmp_path: Path, status: int) -> None:
        class Busy(FakeApi):
            attempts = 0

            def create_repo(self, repo_id: str, **kw: object) -> None:
                self.attempts += 1
                if self.attempts < 3:
                    raise HTTPError(status)

        api = Busy()
        hf.upload(tmp_path, "tester/bookmind", api=api)
        assert api.attempts == 3

    def test_auth_error_mentions_email_verification(self, tmp_path: Path) -> None:
        with pytest.raises(hf.PrepareError, match="email"):
            hf.upload(tmp_path, "tester/bookmind", api=FakeApi(logged_in=False, status=403))

    def test_retry_uses_injected_sleep(self) -> None:
        waits: list[float] = []
        calls = iter([OSError("reset"), "ok"])

        def action() -> object:
            item = next(calls)
            if isinstance(item, Exception):
                raise item
            return item

        assert hf._retry(action, "test", wait_seconds=2, sleep=waits.append) == "ok"
        assert waits == [2]

    def test_main_with_upload(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = _fake_project(tmp_path)
        argv = ["--root", str(root), "--out", str(tmp_path / "s"), "--upload", "tester/bookmind"]
        assert hf.main(argv, api=FakeApi()) == 0
        assert "huggingface.co/spaces/tester/bookmind" in capsys.readouterr().out

    def test_main_upload_errors_return_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _fake_project(tmp_path)
        base = ["--root", str(root), "--out", str(tmp_path / "s"), "--upload"]
        assert hf.main([*base, "tester/bookmind"], api=FakeApi(fail_upload=True)) == 1
        assert "network down" in capsys.readouterr().err
        assert hf.main([*base, "tester/bookmind"], api=FakeApi(logged_in=False)) == 1
        assert "WRITE token" in capsys.readouterr().err


FAKE_HUB = """
import json, os
class HfApi:
    def __init__(self, token):
        self.token = token
    def whoami(self):
        return {"name": "tester"}
    def create_repo(self, repo_id, **kw):
        _log("create_repo", repo_id=repo_id, **kw)
    def upload_folder(self, **kw):
        _log("upload_folder", **kw)
def _log(name, **kw):
    with open(os.environ["FAKE_HUB_LOG"], "a") as fh:
        fh.write(json.dumps([name, kw]) + "\\n")
"""


def _workflow_script(workflow_text: str) -> tuple[dict[str, object], str]:
    """(env of the upload step, the Python program inside its heredoc)."""
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(workflow_text)
    step = next(s for s in doc["jobs"]["deploy"]["steps"] if "HF_TOKEN" in s.get("env", {}))
    run = step["run"]
    program = run.split("<<'PY'\n", 1)[1].rsplit("PY", 1)[0]
    return step["env"], program


class TestGithubWorkflow:
    def test_written_with_space_name(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        out = tmp_path / "space"
        hf.prepare(root, out)
        path = hf.write_github_workflow(root, out, "tester/bookmind")
        assert path == out / ".github" / "workflows" / "deploy-to-hf.yml"
        text = path.read_text(encoding="utf-8")
        assert "{space}" not in text and "tester/bookmind" in text
        assert "${{ secrets.HF_TOKEN }}" in text

    def test_valid_yaml_with_expected_triggers(self, tmp_path: Path) -> None:
        yaml = pytest.importorskip("yaml")
        root = _fake_project(tmp_path)
        text = hf.write_github_workflow(root, tmp_path, "tester/bookmind").read_text()
        doc = yaml.safe_load(text)
        triggers = doc[True] if True in doc else doc["on"]  # YAML 1.1 reads `on` as True
        assert triggers["push"]["branches"] == ["main"] and "workflow_dispatch" in triggers
        env, _ = _workflow_script(text)
        assert env["HF_SPACE"] == "tester/bookmind"

    def test_deploy_script_uploads_folder(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        text = hf.write_github_workflow(root, tmp_path / "o", "tester/bookmind").read_text()
        env, program = _workflow_script(text)
        fake = tmp_path / "fakehub" / "huggingface_hub"
        fake.mkdir(parents=True)
        (fake / "__init__.py").write_text(FAKE_HUB)
        log = tmp_path / "calls.jsonl"
        result = subprocess.run(
            [sys.executable, "-c", program],
            env={
                "PYTHONPATH": str(fake.parent),
                "HF_TOKEN": "hf_test",
                "HF_SPACE": str(env["HF_SPACE"]),
                "GITHUB_SHA": "abcdef123456",
                "FAKE_HUB_LOG": str(log),
            },
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Deployed" in result.stdout and "tester/bookmind" in result.stdout
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert [c[0] for c in calls] == ["create_repo", "upload_folder"]
        assert calls[0][1]["space_sdk"] == "gradio"
        upload = calls[1][1]
        assert upload["repo_id"] == "tester/bookmind" and ".github/*" in upload["ignore_patterns"]
        assert upload["commit_message"].endswith("abcdef1")

    def test_deploy_script_explains_missing_secret(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        _, program = _workflow_script(
            hf.write_github_workflow(root, tmp_path, "tester/bookmind").read_text()
        )
        fake = tmp_path / "fakehub" / "huggingface_hub"
        fake.mkdir(parents=True)
        (fake / "__init__.py").write_text(FAKE_HUB)
        result = subprocess.run(
            [sys.executable, "-c", program],
            env={"PYTHONPATH": str(fake.parent), "HF_SPACE": "tester/bookmind"},
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "HF_TOKEN secret is missing" in result.stderr

    @pytest.mark.parametrize("bad", ["bookmind", "a/b/c", "user/", "my user/bookmind"])
    def test_rejects_bad_space_name(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(hf.PrepareError, match="--github"):
            hf.write_github_workflow(_fake_project(tmp_path), tmp_path, bad)

    def test_main_with_github_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _fake_project(tmp_path)
        argv = ["--root", str(root), "--out", str(tmp_path / "s"), "--github", "tester/bookmind"]
        assert hf.main(argv) == 0
        out = capsys.readouterr().out
        assert ".github/workflows/deploy-to-hf.yml" in out and "HF_TOKEN" in out
        assert (tmp_path / "s" / ".github" / "workflows" / "deploy-to-hf.yml").exists()
        assert (tmp_path / "s" / "app.py").exists()

    def test_main_github_requires_gradio(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _fake_project(tmp_path)
        argv = ["--root", str(root), "--out", str(tmp_path / "s"), "--sdk", "docker"]
        assert hf.main([*argv, "--github", "tester/bookmind"]) == 1
        assert "gradio Space only" in capsys.readouterr().err


class TestTrainedVersions:
    def _art(self, tmp_path: Path, manifest: object) -> Path:
        art = tmp_path / "art"
        art.mkdir()
        if manifest is not None:
            text = manifest if isinstance(manifest, str) else json.dumps(manifest)
            (art / "knn_manifest.json").write_text(text)
        return art

    def test_reads_library_versions(self, tmp_path: Path) -> None:
        art = self._art(tmp_path, {"library_versions": {"pandas": "2.0.0", "numpy": "1.26.0"}})
        assert hf.trained_versions(art) == {"pandas": "2.0.0", "numpy": "1.26.0"}

    def test_old_manifest_falls_back_to_sklearn_version(self, tmp_path: Path) -> None:
        art = self._art(tmp_path, {"sklearn_version": "1.5.0"})
        assert hf.trained_versions(art) == {"scikit-learn": "1.5.0"}

    @pytest.mark.parametrize("manifest", [None, "not json"])
    def test_missing_or_broken_manifest(self, tmp_path: Path, manifest: object) -> None:
        assert hf.trained_versions(self._art(tmp_path, manifest)) == {}

    def test_trained_versions_win_over_installed(self) -> None:
        trained = {"pandas": "2.0.0", "scikit-learn": "1.5.0"}
        text, warnings = hf.gradio_requirements(lookup=FAKE_VERSIONS.get, trained=trained)
        assert "pandas==2.0.0" in text and "scikit-learn==1.5.0" in text
        assert "numpy==2.1.0" in text  # not recorded -> installed version
        assert warnings == []

    def test_ci_warns_when_pickle_versions_were_not_recorded(self) -> None:
        trained = {"scikit-learn": "1.5.0"}
        _, warnings = hf.gradio_requirements(lookup=FAKE_VERSIONS.get, trained=trained, on_ci=True)
        assert {w.split()[0] for w in warnings} == {"numpy", "pandas", "scipy"}
        assert all("train-knn" in w for w in warnings)

    def test_prepare_warns_on_github_actions_without_recorded_versions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        warnings = hf.prepare(_fake_project(tmp_path), tmp_path / "space")
        assert sorted(w.split()[0] for w in warnings) == [
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
        ]

    def test_prepare_on_github_actions_is_quiet_with_recorded_versions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        root = _fake_project(tmp_path)
        recorded = {"numpy": "2.1.0", "pandas": "2.2.3", "scikit-learn": "1.6.1", "scipy": "1.15.0"}
        (root / "artifacts" / "knn_manifest.json").write_text(
            json.dumps({"library_versions": recorded})
        )
        assert hf.prepare(root, tmp_path / "space") == []

    def test_prepare_uses_manifest_versions(self, tmp_path: Path) -> None:
        root = _fake_project(tmp_path)
        (root / "artifacts" / "knn_manifest.json").write_text(
            json.dumps({"library_versions": {"pandas": "2.0.0"}})
        )
        hf.prepare(root, tmp_path / "space")
        reqs = (tmp_path / "space" / "requirements.txt").read_text(encoding="utf-8")
        assert "pandas==2.0.0" in reqs and "numpy==2.1.0" in reqs


class TestDeployWorkflow:
    """The repository's own .github/workflows/deploy.yml (continuous deployment)."""

    def _doc(self) -> dict[str, object]:
        yaml = pytest.importorskip("yaml")
        text = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
        doc: dict[str, object] = yaml.safe_load(text)
        return doc

    def test_runs_after_successful_ci_on_main(self) -> None:
        doc = self._doc()
        triggers = doc.get(True) or doc["on"]  # YAML 1.1 reads `on` as True
        assert isinstance(triggers, dict)
        run = triggers["workflow_run"]
        assert run["workflows"] == ["CI"] and run["branches"] == ["main"]
        assert "workflow_dispatch" in triggers
        job = doc["jobs"]["hugging-face"]  # type: ignore[index]
        assert "conclusion == 'success'" in job["if"]

    def test_ci_workflow_name_matches(self) -> None:
        yaml = pytest.importorskip("yaml")
        ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
        assert ci["name"] == "CI"

    def test_uses_prepare_script_with_upload_and_secret(self) -> None:
        job = self._doc()["jobs"]["hugging-face"]  # type: ignore[index]
        assert job["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"
        runs = " ".join(step.get("run", "") for step in job["steps"])
        assert "scripts/prepare_hf_space.py --upload" in runs
        assert "huggingface_hub" in runs and "packaging" in runs

    def test_artifacts_are_not_gitignored(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "artifacts/*" not in ignore
        assert "hf_space/" in ignore and ".venv/" in ignore
