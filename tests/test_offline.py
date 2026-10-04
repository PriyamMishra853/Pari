"""The offline guarantee, asserted rather than promised.

The demo opens by unplugging the ethernet cable and saying "everything runs on
this box". PLAN.md section 15 attack 5 answers a certification question with
"no network path exists in the inference code". Both are claims about the
source, so both can be checked by reading it - and a claim that is checked in CI
survives someone adding a convenient telemetry call in week 10.

Two separate assertions:

  The inference layers - belief, pdl, engine, perception, eval - import nothing
  that can open a socket. Not "we do not call out", but "the capability is not
  linked in".

  io/ has exactly ONE outbound path, `CaptureConfig.stream_url`, and it is
  configured, explicit, and off by default. That is the ground-station stream
  the problem statement asks for; it is not a dependency of anything.

This is also a privacy argument. Video never leaves the box unless someone
configures it to, so DPDP-alignment is a property of the architecture rather
than of a policy document.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "parikshak"

#: Anything that can open a socket. Not exhaustive against determined effort -
#: it is a guard against drift, not a sandbox.
NETWORK_MODULES = frozenset({
    "socket", "socketserver", "ssl", "select", "selectors",
    "http", "urllib", "urllib2", "urllib3", "ftplib", "poplib", "imaplib",
    "smtplib", "telnetlib", "nntplib", "xmlrpc", "webbrowser",
    "requests", "httpx", "aiohttp", "websockets", "websocket",
    "paramiko", "boto3", "botocore", "google", "openai", "anthropic",
    "pika", "kafka", "redis", "pymongo", "psycopg2", "grpc",
})

#: Layers that must have no network capability whatsoever.
INFERENCE_LAYERS = ("belief", "pdl", "engine", "perception", "eval")


def imported(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def sources(*layers: str) -> list[Path]:
    if not layers:
        return sorted(PKG.rglob("*.py"))
    return sorted(p for layer in layers for p in (PKG / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", INFERENCE_LAYERS)
def test_inference_layers_cannot_reach_the_network(layer):
    """Not "does not call out" - cannot. The capability is not imported."""
    offences = []
    for src in sources(layer):
        for module in imported(src) & NETWORK_MODULES:
            offences.append(f"{src.relative_to(PKG.parent)} imports {module}")
    assert not offences, (
        "the inference path must run with the cable unplugged:\n  " + "\n  ".join(offences))


def test_the_whole_package_has_no_network_imports():
    """Including io/. The one outbound path is a GStreamer sink built as a
    string and handed to GStreamer - no Python socket is ever opened here."""
    offences = []
    for src in sources():
        for module in imported(src) & NETWORK_MODULES:
            offences.append(f"{src.relative_to(PKG.parent)} imports {module}")
    assert not offences, "\n  ".join(offences)


def test_no_urls_are_hardcoded_in_the_inference_path():
    """A URL in the inference layers is either a dead constant or a call
    waiting to happen. Neither belongs there."""
    offences = []
    for src in sources(*INFERENCE_LAYERS):
        text = src.read_text(encoding="utf-8")
        for marker in ("http://", "https://", "rtsp://", "srt://", "ws://"):
            if marker in text:
                offences.append(f"{src.relative_to(PKG.parent)} contains {marker!r}")
    assert not offences, "\n  ".join(offences)


def test_the_only_outbound_path_is_the_configured_stream():
    """One place, named, and off unless someone sets it."""
    from parikshak.io.capture import CaptureConfig, build_pipeline

    assert CaptureConfig().stream_url is None
    quiet = build_pipeline(CaptureConfig(record_dir=Path("/tmp/rec")))
    assert "://" not in quiet

    loud = build_pipeline(CaptureConfig(record_dir=Path("/tmp/rec"),
                                        stream_url="rtsp://10.0.0.5:8554/rack"))
    assert loud.count("://") == 1


def test_subprocess_use_is_limited_to_local_media_tools():
    """ffmpeg for clip cutting, piper for speech, nvidia-smi / the OS GPU query
    for hardware detection - all local binaries. A subprocess elsewhere in the
    package would be an unreviewed escape hatch."""
    allowed = {"parikshak/io/clips.py", "parikshak/io/tts.py", "parikshak/zerog/hardware.py"}
    offences = []
    for src in sources():
        rel = src.relative_to(PKG.parent).as_posix()
        if rel in allowed:
            continue
        if "subprocess" in imported(src):
            offences.append(rel)
    assert not offences, f"unexpected subprocess use: {offences}"


def test_optional_extras_are_never_imported_at_module_scope():
    """The engine and the eval harness must run on a laptop with no vision or
    audio stack. A top-level `import mediapipe` anywhere in the package would
    make importing parikshak fail on exactly the machine most work happens on.
    """
    heavy = {"cv2", "mediapipe", "onnxruntime", "pupil_apriltags", "vosk",
             "gi", "piper", "PySide6", "sounddevice"}
    offences = []
    for src in sources():
        tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
        for node in tree.body:                      # module scope only
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in heavy:
                        offences.append(f"{src.relative_to(PKG.parent)}: {a.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in heavy:
                    offences.append(f"{src.relative_to(PKG.parent)}: {node.module}")
    assert not offences, "heavy imports at module scope:\n  " + "\n  ".join(offences)


def test_importing_the_package_needs_nothing_optional():
    """The practical version of the test above."""
    import importlib

    for module in ("parikshak.belief", "parikshak.pdl", "parikshak.engine.runner",
                   "parikshak.perception", "parikshak.eval.harness", "parikshak.io",
                   "parikshak.gui"):
        importlib.import_module(module)
