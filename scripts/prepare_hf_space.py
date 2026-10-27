"""
Assemble a ready-to-upload Hugging Face Space folder.

    python scripts/prepare_hf_space.py              # Gradio Space (free) -> ./hf_space
    python scripts/prepare_hf_space.py --sdk docker # Docker Space (paid on HF) -> ./hf_space
    python scripts/prepare_hf_space.py --upload <hf-username>/bookmind
                                                    # prepare + create the Space + upload
    python scripts/prepare_hf_space.py --github <hf-username>/bookmind
                                                    # + a GitHub Actions workflow that
                                                    #   deploys to the Space from GitHub

Gradio (default, free): app.py from deploy/gradio/, a README card and a requirements.txt
pinned to the library versions on THIS computer (the k-NN pickles need the same pandas /
scikit-learn they were trained with), the service-layer packages of `bookmind`, and the
pre-built artifacts/. Docker: the full FastAPI app with deploy/huggingface/Dockerfile.
Upload the *contents* of the folder to your Space (see docs/DEPLOY_HUGGINGFACE.md).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_ARTIFACTS = ("book_index.faiss", "book_meta.parquet", "manifest.json")
KNN_ARTIFACTS = (
    "model_knn.pkl",
    "book_names.pkl",
    "final_rating.pkl",
    "book_pivot.pkl",
    "knn_manifest.json",
)
PROJECT_FILES = ("pyproject.toml", "requirements.txt", "gunicorn.conf.py")  # docker only
SPACE_FILES = ("Dockerfile", "README.md")  # docker only, from deploy/huggingface/

# ── Gradio Space ─────────────────────────────────────────────────────────────
GRADIO_SDK_VERSION = "6.30.0"  # the version the Gradio app was tested with
SPACE_PYTHON = "3.12"  # supported by Hugging Face ZeroGPU hardware
GRADIO_PACKAGES = ("core", "ml", "schemas", "services")  # all that app.py imports
# Pickled k-NN files must be read by the same versions that wrote them -> exact pins.
EXACT_PINS = ("numpy", "pandas", "scikit-learn", "scipy")
OTHER_PINS = (
    "faiss-cpu",
    "pyarrow",
    "sentence-transformers",
    "pydantic",
    "pydantic-settings",
)
FALLBACK_REQUIREMENTS = {  # used when a library is not installed locally
    "numpy": "numpy>=1.26,<3.0",
    "pandas": "pandas>=2.2,<4.0",
    "scikit-learn": "scikit-learn>=1.4,<2.0",
    "scipy": "scipy>=1.12,<2.0",
    "faiss-cpu": "faiss-cpu>=1.8,<2.0",
    "pyarrow": "pyarrow>=15.0",
    "sentence-transformers": "sentence-transformers>=3.0,<6.0",
    "pydantic": "pydantic>=2.7,<3.0",
    "pydantic-settings": "pydantic-settings>=2.3,<3.0",
}
HF_FILE_LIMIT_MB = 10  # bigger files are stored with Git LFS (the web uploader does this)


class PrepareError(Exception):
    """Something the user must fix before the Space can be built."""


def _mb(path: Path) -> float:
    return path.stat().st_size / 1_048_576


def check_artifacts(artifacts: Path) -> tuple[list[str], list[str]]:
    """Return (files to copy, warnings). Raise PrepareError if the index is unusable."""
    missing = [f for f in REQUIRED_ARTIFACTS if not (artifacts / f).exists()]
    if missing:
        raise PrepareError(
            f"artifacts/ is missing {', '.join(missing)}. Build the index first:\n"
            "  python -m bookmind.pipeline.cli all --classifier keyword"
        )
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    backend = manifest.get("embedding_backend")
    if backend != "sentence-transformers":
        raise PrepareError(
            f"The index was built with the '{backend}' encoder. The Space uses "
            "sentence-transformers - rebuild with:\n"
            "  python -m bookmind.pipeline.cli build-index --backend sentence-transformers"
        )

    files = list(REQUIRED_ARTIFACTS)
    warnings: list[str] = []
    present_knn = [f for f in KNN_ARTIFACTS if (artifacts / f).exists()]
    if len(present_knn) == len(KNN_ARTIFACTS):
        files += present_knn
    else:
        warnings.append(
            "k-NN files not found - the Space will use semantic similarity only. "
            "Run `python -m bookmind.pipeline.cli train-knn` to include them."
        )
    return files, warnings


PROTECTED = {"src", "tests", "docs", "data", "artifacts", "scripts", "deploy", ".venv", ".git"}


def _check_out(root: Path, out: Path) -> None:
    """Refuse output folders whose deletion would damage the project."""
    r, o = root.resolve(), out.resolve()
    if o == r or o in r.parents:
        raise PrepareError("The output folder must not be the project folder or its parent.")
    if r in o.parents and (o.parent != r or o.name in PROTECTED):
        raise PrepareError(
            "Inside the project, only a new top-level folder (default: hf_space) can be used."
        )


def _installed(package: str) -> tuple[str, str | None] | None:
    """(version, Requires-Python) of a locally installed distribution, or None."""
    try:
        from importlib.metadata import PackageNotFoundError, metadata
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        meta = metadata(package)
    except PackageNotFoundError:
        return None
    return meta["Version"], meta.get("Requires-Python")


def _supports_python(requires_python: str | None, python: str) -> bool:
    if not requires_python:
        return True
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:  # pragma: no cover - packaging ships with pip/setuptools
        return True
    return SpecifierSet(requires_python).contains(f"{python}.0")


def trained_versions(artifacts: Path) -> dict[str, str]:
    """Library versions recorded by `train-knn` in knn_manifest.json (may be partial/empty)."""
    path = artifacts / "knn_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    versions = {k: str(v) for k, v in (data.get("library_versions") or {}).items()}
    if "scikit-learn" not in versions and data.get("sklearn_version"):
        versions["scikit-learn"] = str(data["sklearn_version"])
    return versions


def gradio_requirements(
    lookup: Callable[[str], tuple[str, str | None] | None] = _installed,
    python: str = SPACE_PYTHON,
    trained: dict[str, str] | None = None,
    on_ci: bool = False,
) -> tuple[str, list[str]]:
    """requirements.txt text for the Space, plus warnings.

    Pickle-sensitive libraries are pinned to the versions recorded at training time
    (knn_manifest.json); otherwise to the versions installed on this computer.
    """
    trained = trained or {}
    lines = [
        "# Generated by scripts/prepare_hf_space.py from the versions on the build computer.",
        "# gradio itself comes from sdk_version in README.md; torch is pulled in by",
        "# sentence-transformers.",
    ]
    warnings: list[str] = []
    for package in (*EXACT_PINS, *OTHER_PINS):
        if package in trained:
            lines.append(f"{package}=={trained[package]}")
            continue
        found = lookup(package)
        if package in EXACT_PINS and on_ci:
            warnings.append(
                f"{package} version was not recorded at training time, so it is pinned to this "
                "build server's version. Re-run `train-knn` locally and commit the artifacts to "
                "record it."
            )
        if found is None:
            lines.append(FALLBACK_REQUIREMENTS[package])
            if package in EXACT_PINS:
                warnings.append(
                    f"{package} is not installed here, so it is not pinned; the k-NN files "
                    "may not load if the Space installs a different version."
                )
            continue
        version, requires_python = found
        if _supports_python(requires_python, python):
            lines.append(f"{package}=={version}")
        else:
            lines.append(FALLBACK_REQUIREMENTS[package])
            warnings.append(
                f"{package} {version} does not support Python {python} (needs "
                f"{requires_python}); the Space installs a compatible version instead."
            )
    return "\n".join(lines) + "\n", warnings


SHORT_DESCRIPTION_LIMIT = 60  # Hugging Face rejects longer Space descriptions


def check_space_card(text: str) -> None:
    """Catch Space README metadata that Hugging Face would reject, before uploading."""
    if not text.startswith("---"):
        raise PrepareError("Space README.md must start with a '---' metadata block.")
    meta = text.split("---", 2)[1]
    match = re.search(r"^short_description:\s*(.+)$", meta, flags=re.MULTILINE)
    if match and len(match.group(1).strip()) > SHORT_DESCRIPTION_LIMIT:
        raise PrepareError(
            f"short_description in the Space README is {len(match.group(1).strip())} characters; "
            f"Hugging Face allows at most {SHORT_DESCRIPTION_LIMIT}."
        )
    if "{" in meta:
        raise PrepareError("Space README.md metadata still contains a {placeholder}.")
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is installed with pre-commit / gradio
        return
    try:
        yaml.safe_load(meta)
    except yaml.YAMLError as exc:
        raise PrepareError(f"Space README.md metadata is not valid YAML: {exc}") from exc


def _copy_artifacts(artifacts: Path, dest: Path, files: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(artifacts / name, dest / name)


def _zip_package(package: Path, target: Path) -> None:
    """bookmind/__init__.py + the service-layer packages, as an importable zip."""
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(package / "__init__.py", "bookmind/__init__.py")
        for sub in GRADIO_PACKAGES:
            for path in sorted((package / sub).rglob("*.py")):
                zf.write(path, f"bookmind/{path.relative_to(package).as_posix()}")


def _prepare_gradio(root: Path, out: Path, artifact_files: list[str]) -> list[str]:
    """Flat folder (no sub-folders) so the browser uploader accepts it in one drop."""
    src = root / "deploy" / "gradio"
    shutil.copy2(src / "app.py", out / "app.py")
    card = (src / "README.md").read_text(encoding="utf-8")
    card = card.replace("{sdk_version}", GRADIO_SDK_VERSION).replace(
        "{python_version}", SPACE_PYTHON
    )
    check_space_card(card)
    (out / "README.md").write_text(card, encoding="utf-8")
    text, warnings = gradio_requirements(
        lookup=_installed,
        trained=trained_versions(root / "artifacts"),
        on_ci=bool(os.environ.get("GITHUB_ACTIONS")),
    )
    (out / "requirements.txt").write_text(text, encoding="utf-8")
    _zip_package(root / "src" / "bookmind", out / "bookmind.zip")
    _copy_artifacts(root / "artifacts", out, artifact_files)
    return warnings


def _check_repo_id(repo_id: str, option: str) -> None:
    if repo_id.count("/") != 1 or not all(repo_id.split("/")) or " " in repo_id:
        raise PrepareError(f"{option} expects <hf-username>/<space-name>, e.g. yogita/bookmind")


def write_github_workflow(root: Path, out: Path, space: str) -> Path:
    """Add .github/workflows/deploy-to-hf.yml that uploads the folder to ``space``."""
    _check_repo_id(space, "--github")
    template = (root / "deploy" / "gradio" / "deploy-to-hf.yml").read_text(encoding="utf-8")
    target = out / ".github" / "workflows" / "deploy-to-hf.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("{space}", space), encoding="utf-8")
    return target


def _prepare_docker(root: Path, out: Path, artifact_files: list[str]) -> list[str]:
    for name in PROJECT_FILES:
        shutil.copy2(root / name, out / name)
    check_space_card((root / "deploy" / "huggingface" / "README.md").read_text(encoding="utf-8"))
    for name in SPACE_FILES:
        shutil.copy2(root / "deploy" / "huggingface" / name, out / name)
    shutil.copytree(
        root / "src",
        out / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    _copy_artifacts(root / "artifacts", out / "artifacts", artifact_files)
    return []


def prepare(root: Path, out: Path, sdk: str = "gradio") -> list[str]:
    """Create the Space folder at ``out``. Returns warnings to show the user."""
    if sdk not in ("gradio", "docker"):
        raise PrepareError(f"Unknown --sdk {sdk!r}; use gradio or docker.")
    files, warnings = check_artifacts(root / "artifacts")

    _check_out(root, out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    builder = _prepare_gradio if sdk == "gradio" else _prepare_docker
    return warnings + builder(root, out, files)


def _is_auth_error(exc: BaseException) -> bool:
    """True for a missing/invalid/read-only token (HTTP 401/403 or no saved token)."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403) or type(exc).__name__ == "LocalTokenNotFoundError"


def _is_client_error(exc: BaseException) -> bool:
    """4xx answers other than timeouts/rate limits: the request itself is wrong."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 429)


_NETWORK_NAMES = ("Connect", "Timeout", "Network", "RemoteProtocol", "ReadError", "Reset")


def _is_network_error(exc: BaseException) -> bool:
    """Dropped connections, timeouts and temporary server errors: worth retrying."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in (408, 429) or status >= 500
    return any(part in type(exc).__name__ for part in _NETWORK_NAMES)


def _retry(
    action: Callable[[], object],
    what: str,
    attempts: int = 3,
    wait_seconds: float = 5.0,
    sleep: Callable[[float], None] | None = None,
) -> object:
    """Run a Hugging Face call; retry network drops, explain login problems clearly."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:
            if _is_auth_error(exc):
                raise PrepareError(
                    f"Hugging Face refused the request: {exc}\n"
                    "Usual causes: the token is missing, invalid or READ-only, or your email "
                    "is not verified yet (check your inbox). Log in again with a WRITE token:\n"
                    '  python -c "import getpass; from huggingface_hub import login; '
                    "login(token=getpass.getpass('Token: '))\""
                ) from exc
            if _is_client_error(exc) or not _is_network_error(exc):
                # e.g. 400 bad name or invalid README metadata: retrying will not help
                raise PrepareError(f"Hugging Face could not {what}: {exc}") from exc
            if attempt == attempts:
                raise PrepareError(
                    f"Could not {what}: {type(exc).__name__}: {exc}\n"
                    "This is a network problem, not your code. Check https://huggingface.co "
                    "opens in the browser, switch off VPN/proxy or try a phone hotspot, "
                    "then run the same command again - or upload the folder in the browser "
                    "(docs/DEPLOY_HUGGINGFACE.md)."
                ) from exc
            print(
                f"[hf-space] Problem ({type(exc).__name__}: {str(exc)[:160]}) - retrying "
                f"in {wait_seconds:.0f}s ({attempt}/{attempts - 1}) …"
            )
            (sleep or time.sleep)(wait_seconds)
    raise AssertionError("unreachable")  # pragma: no cover


def upload(out: Path, repo_id: str, api: Any | None = None, sdk: str = "gradio") -> str:
    """Create the Space if needed and upload the folder. Returns the Space URL."""
    _check_repo_id(repo_id, "--upload")
    if api is None:  # pragma: no cover - real network client
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            msg = "Install the client first: python -m pip install huggingface_hub"
            raise PrepareError(msg) from exc
        api = HfApi()
    _retry(api.whoami, "check your login")
    _retry(
        lambda: api.create_repo(repo_id, repo_type="space", space_sdk=sdk, exist_ok=True),
        "create the Space",
    )
    _retry(
        lambda: api.upload_folder(
            folder_path=str(out),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy BookMind Pro",
            # remove code/artifact files that no longer exist locally
            delete_patterns=["src/**", "bookmind/**", "artifacts/**", "*.pkl", "*.faiss"],
        ),
        "upload the files",
    )
    return f"https://huggingface.co/spaces/{repo_id}"


def main(argv: list[str] | None = None, api: Any | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=ROOT / "hf_space", help="Output folder")
    parser.add_argument(
        "--sdk",
        choices=("gradio", "docker"),
        default="gradio",
        help="gradio = free Space (default); docker = full FastAPI app (paid on Hugging Face)",
    )
    parser.add_argument(
        "--upload", metavar="USER/SPACE", help="Also create/update this Hugging Face Space"
    )
    parser.add_argument(
        "--github",
        metavar="USER/SPACE",
        help="Also add a GitHub Actions workflow that deploys to this Space (gradio only)",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        if args.github and args.sdk != "gradio":
            raise PrepareError("--github works with the gradio Space only.")
        warnings = prepare(args.root, args.out, sdk=args.sdk)
        workflow = write_github_workflow(args.root, args.out, args.github) if args.github else None
    except PrepareError as exc:
        print(f"[hf-space] {exc}", file=sys.stderr)
        return 1

    print(f"[hf-space] Ready ({args.sdk} Space): {args.out}")
    total = 0.0
    for path in sorted(args.out.rglob("*")):
        if path.is_file():
            size = _mb(path)
            total += size
            if path.parent == args.out or path.parent.name == "artifacts":
                flag = "  (stored with Git LFS)" if size > HF_FILE_LIMIT_MB else ""
                name = path.relative_to(args.out).as_posix()
                print(f"    {name:<32} {size:7.2f} MB{flag}")
    print(f"[hf-space] Total size: {total:.1f} MB")
    for warning in warnings:
        print(f"[hf-space] WARNING: {warning}")
    if workflow is not None:
        print(f"    {workflow.relative_to(args.out).as_posix():<32} (GitHub deploy workflow)")
        print("[hf-space] Next: upload ALL of this folder, including the .github folder, to a")
        print("           GitHub repository; add the HF_TOKEN secret; GitHub deploys it to")
        print(f"           https://huggingface.co/spaces/{args.github}")
        print("           Step-by-step guide: docs/DEPLOY_HUGGINGFACE.md (Route B)")
    if not args.upload:
        if workflow is None:
            what = "ALL files listed above" if args.sdk == "gradio" else "the folder's CONTENTS"
            print(f"[hf-space] Next: select {what} and drag them into your Space in the browser")
            print(
                "           (Files -> Contribute -> Upload files), or add --upload <user>/bookmind."
            )
            print("           Step-by-step guide: docs/DEPLOY_HUGGINGFACE.md")
        return 0

    print(f"[hf-space] Uploading to {args.upload} … (first upload can take a few minutes)")
    try:
        url = upload(args.out, args.upload, api=api, sdk=args.sdk)
    except PrepareError as exc:
        print(f"[hf-space] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # network / permission errors from the hub
        print(f"[hf-space] Upload failed: {exc}", file=sys.stderr)
        return 1
    print(f"[hf-space] Uploaded. Build logs and app: {url}")
    print("           The first build takes about 5-10 minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
