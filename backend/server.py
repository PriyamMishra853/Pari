"""PARIKSHAK mission server.

Pages
  /              landing page
  /console       mission workstation (live camera, procedure, copilot, 3D rack world)
  /presentation  SIH deck
  /tags          printable AprilTag sheet for the rack frame

API (old /api/tracker/* and /api/guide/* paths are kept as aliases)
  /api/system, /api/experiments, /api/frame, /api/telemetry, /api/copilot/*, /api/voice/*,
  /api/video/*, /api/rotation_test/*, /api/dataset/*, /api/rack/*, recordings, reports
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from starlette.requests import Request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo import scenarios as S  # noqa: E402

FRONTEND_DIR = ROOT / "frontend"
LANDING_PAGE = FRONTEND_DIR / "index.html"
CONSOLE_PAGE = FRONTEND_DIR / "console.html"


def get_landing_page() -> str:
    if LANDING_PAGE.exists():
        return LANDING_PAGE.read_text(encoding="utf-8")
    return "<!doctype html><title>PARIKSHAK</title><h1>PARIKSHAK</h1>"


def get_console_page(bundle: dict[str, Any] | None = None) -> str:
    body = CONSOLE_PAGE.read_text(encoding="utf-8") if CONSOLE_PAGE.exists() else "<h1>Console not found</h1>"
    if bundle is not None:
        data = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
        body = body.replace("<!--BUNDLE-->", f"<script>window.PARIKSHAK_BUNDLE={data};</script>")
    return body


def _tags_page(size_m: float) -> str:
    from parikshak.zerog.config import rack_layout

    lay = rack_layout()
    cards = "".join(
        f'<figure><img src="/api/tags/{i}.png" alt="AprilTag {i}"><figcaption>tag36h11 &middot; ID {i}'
        f'<br><small>{lay.labels.get(i, "")} &middot; rack position {list(map(float, lay.tags[i]))} m</small></figcaption></figure>'
        for i in sorted(lay.tags)
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rack AprilTags</title><style>
:root{{--bg:#fff;--fg:#0f172a;--muted:#475569}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0b1020;--fg:#e2e8f0;--muted:#94a3b8}}}}
body{{margin:0;padding:24px 16px;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:960px;margin:auto}} h1{{margin:.2em 0}} p,li{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px;margin-top:18px}}
figure{{margin:0;background:#fff;color:#0f172a;border:1px solid #cbd5e1;border-radius:10px;padding:14px;text-align:center}}
figure img{{width:100%;image-rendering:pixelated}} small{{color:#475569}}
@media print{{body{{background:#fff;color:#000}} .noprint{{display:none}} figure{{break-inside:avoid}}}}
</style></head><body><main>
<h1>Rack fiducials (AprilTag 36h11)</h1>
<div class="noprint"><p>The rack is the reference frame. Fix <b>tag 10</b> upright on the rack face (a wall, a box front, the container)
so the camera sees it together with you. One tag gives the full 6-DoF rack pose; more tags make it steadier.</p>
<ol><li>Print this page (or show tag 10 full-screen on a phone or tablet).</li>
<li>Measure the black square's side and enter it as <b>Tag size</b> in the console (configured now: {size_m * 100:.1f} cm).</li>
<li>Keep the tag flat, upright and well lit. Rotating the laptop is fine - the rack frame stays the reference.</li></ol></div>
<div class="grid">{cards}</div></main></body></html>"""


def create_app():
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles

    from backend.app.groq.client import get_client
    from backend.voice import COMMANDS, LABELS, match, whisper_prompt
    from parikshak.perception.tracker_service import get_tracker_service
    from parikshak.zerog.dataset import DatasetRecorder

    app = FastAPI(title="PARIKSHAK Mission Server", docs_url="/api/docs", redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    svc = get_tracker_service()

    for name in ("static", "recordings", "reports"):
        p = (FRONTEND_DIR / "static") if name == "static" else (ROOT / name)
        p.mkdir(parents=True, exist_ok=True)
        app.mount(f"/{name}", StaticFiles(directory=str(p)), name=name)

    async def body(req: Request) -> dict[str, Any]:
        try:
            return await req.json()
        except Exception:
            return {}

    # ------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return get_landing_page()

    @app.get("/console", response_class=HTMLResponse)
    @app.get("/app", response_class=HTMLResponse)
    @app.get("/workspace", response_class=HTMLResponse)
    def console() -> str:
        return get_console_page()

    @app.get("/presentation", response_class=HTMLResponse)
    def presentation_page():
        for p in (FRONTEND_DIR / "sih_presentation.html", ROOT / "demo" / "static" / "sih_presentation.html"):
            if p.exists():
                return HTMLResponse(p.read_text(encoding="utf-8"))
        raise HTTPException(404, "Presentation page not found")

    @app.get("/tags", response_class=HTMLResponse)
    def tags_page() -> str:
        return _tags_page(svc.pipeline.rack.tag_size_m)

    @app.get("/api/tags/{tag_id}.png")
    def tag_png(tag_id: int, px: int = 600):
        import cv2
        import numpy as np

        if not (0 <= tag_id < 587):
            raise HTTPException(404, "tag36h11 ids are 0-586")
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        px = max(100, min(1600, px))
        m = cv2.aruco.generateImageMarker(d, tag_id, px)
        q = px // 8
        canvas = np.full((px + 2 * q, px + 2 * q), 255, np.uint8)
        canvas[q:q + px, q:q + px] = m
        ok, buf = cv2.imencode(".png", canvas)
        return Response(buf.tobytes(), media_type="image/png")

    @app.get("/api/download_pptx")
    def download_pptx():
        p = ROOT / "PARIKSHAK_SIH2026_Submission.pptx"
        if p.exists():
            return FileResponse(str(p), filename=p.name,
                                media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")
        raise HTTPException(404, "Presentation file not found")

    # ------------------------------------------------- scenario replays
    @app.get("/api/scenarios")
    def list_scenarios() -> dict[str, Any]:
        return {"default": S.DEFAULT, "groups": S.GROUP_ORDER, "scenarios": S.catalogue()}

    @app.get("/api/scenario/{scenario_id}")
    def get_scenario(scenario_id: str) -> dict[str, Any]:
        try:
            return S.replay(scenario_id)
        except KeyError:
            raise HTTPException(404, f"unknown scenario {scenario_id!r}")

    @app.get("/api/results")
    def get_results() -> dict[str, Any]:
        def rd(n):
            p = ROOT / "runs" / n
            return json.loads(p.read_text("utf-8")) if p.exists() else None
        return {"csp1": rd("eval.json"), "crx2": rd("eval_crx2.json"), "bench": rd("bench.json"), "soak": rd("soak.json")}

    # ------------------------------------------------------------ system
    @app.get("/api/system")
    def system() -> dict[str, Any]:
        return svc.system_status()

    @app.post("/api/mesh_backend")
    async def mesh_backend(req: Request) -> dict[str, Any]:
        return svc.set_mesh_backend(str((await body(req)).get("backend", "pose3d")))

    # ------------------------------------------------------- experiments
    @app.get("/api/experiments")
    @app.get("/api/tracker/experiments")
    def experiments() -> dict[str, Any]:
        return svc.get_experiments_list()

    @app.get("/api/experiments/object_classes")
    def object_classes() -> dict[str, Any]:
        from parikshak.zerog.procedure import PREDICATES, TEXT

        return {"classes": svc.object_classes(),
                "checks": {k: TEXT.get(k, k) for k in sorted(PREDICATES)}}

    @app.post("/api/experiments/select")
    @app.post("/api/tracker/set_experiment")
    async def select_experiment(req: Request) -> dict[str, Any]:
        return svc.set_experiment(str((await body(req)).get("experiment_id", "WBP-1")))

    @app.post("/api/tracker/experiment/{experiment_id}")
    def select_experiment_path(experiment_id: str) -> dict[str, Any]:
        return svc.set_experiment(experiment_id)

    @app.post("/api/experiments/custom")
    @app.post("/api/experiment/create_custom")
    async def create_custom(req: Request) -> dict[str, Any]:
        res = svc.create_custom_experiment(await body(req))
        if "error" in res:
            raise HTTPException(400, res["error"])
        return res

    # ------------------------------------------------------------ frames
    @app.post("/api/frame")
    @app.post("/api/tracker/frame")
    async def frame(req: Request) -> dict[str, Any]:
        data = await body(req)
        img = data.get("image") or data.get("frame") or ""
        if not img:
            raise HTTPException(400, "image (base64 JPEG) required")
        return svc.process_b64_frame(img)

    @app.get("/api/telemetry")
    @app.get("/api/tracker/telemetry")
    def telemetry() -> dict[str, Any]:
        return svc.get_telemetry()

    @app.post("/api/reset")
    @app.post("/api/tracker/reset")
    @app.post("/api/tracker/restart")
    def reset() -> dict[str, Any]:
        return svc.reset()

    @app.post("/api/tracker/toggles")
    async def toggles(req: Request) -> dict[str, Any]:
        return {"toggles": svc.set_toggles(await body(req))}

    # ----------------------------------------------------------- copilot
    @app.post("/api/copilot/review")
    @app.post("/api/guide/review_step")
    def copilot_review() -> dict[str, Any]:
        return svc.copilot_action("review")

    @app.post("/api/copilot/next")
    @app.post("/api/guide/convey_next")
    def copilot_next() -> dict[str, Any]:
        return svc.copilot_action("next")

    @app.get("/api/copilot/status_line")
    def copilot_status() -> dict[str, Any]:
        return svc.status_line()

    @app.get("/api/copilot/feed")
    def copilot_feed(after: int = 0) -> dict[str, Any]:
        return {"feed": svc.copilot.since(after)}

    @app.post("/api/copilot/auto")
    async def copilot_auto(req: Request) -> dict[str, Any]:
        svc.copilot.auto_ai = bool((await body(req)).get("enabled", True))
        return {"auto_ai": svc.copilot.auto_ai}

    # ------------------------------------------------------------- voice
    @app.get("/api/voice/commands")
    def voice_commands() -> dict[str, Any]:
        return {"commands": COMMANDS, "labels": LABELS, "stt": get_client().status()}

    @app.post("/api/voice/command")
    async def voice_command(file: UploadFile = File(...)) -> dict[str, Any]:
        audio = await file.read()
        if len(audio) < 1200:
            return {"transcript": "", "command": None, "error": "audio too short"}
        text, meta = get_client().transcribe(audio, file.filename or "voice.webm", whisper_prompt())
        if text is None:
            return {"transcript": None, "command": None, **meta}
        m = match(text)
        return {"transcript": text, **m, **meta}

    # ------------------------------------------------------------- video
    @app.post("/api/video/upload")
    @app.post("/api/tracker/upload_video")
    async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
        up = ROOT / "runs" / "uploads"
        up.mkdir(parents=True, exist_ok=True)
        name = Path(file.filename or "upload.mp4").name
        p = up / name
        p.write_bytes(await file.read())
        res = svc.load_video_file(p)
        res["filename"] = name
        return res

    @app.post("/api/video/recording/{name}")
    def play_recording(name: str) -> dict[str, Any]:
        p = ROOT / "recordings" / Path(name).name
        return svc.load_video_file(p)

    @app.get("/api/video/next")
    @app.get("/api/tracker/video_frame")
    @app.get("/api/tracker/next_video_frame")
    def video_next() -> dict[str, Any]:
        return svc.get_next_video_frame()

    @app.post("/api/video/stop")
    def video_stop() -> dict[str, Any]:
        return svc.stop_video()

    # ------------------------------------------------------ rotation test
    @app.post("/api/rotation_test/start")
    def rot_start() -> dict[str, Any]:
        return svc.pipeline.start_rotation_test()

    @app.post("/api/rotation_test/stop")
    def rot_stop() -> dict[str, Any]:
        return svc.pipeline.stop_rotation_test()

    @app.get("/api/rotation_test")
    @app.get("/api/tracker/rotation_test")
    def rot_report() -> dict[str, Any]:
        return svc.pipeline.rotation_report()

    @app.post("/api/rack/tag_size")
    async def tag_size(req: Request) -> dict[str, Any]:
        size = float((await body(req)).get("size_m", 0.08))
        svc.pipeline.rack.set_tag_size(size)
        return {"tag_size_m": svc.pipeline.rack.tag_size_m}

    # ----------------------------------------------------------- dataset
    @app.post("/api/dataset/start")
    async def dataset_start(req: Request) -> dict[str, Any]:
        every = int((await body(req)).get("every", 3))
        return svc.dataset.start(svc.experiment_id, svc.spec, every)

    @app.post("/api/dataset/stop")
    def dataset_stop() -> dict[str, Any]:
        return svc.dataset.stop()

    @app.get("/api/dataset")
    def dataset_list() -> dict[str, Any]:
        return {"status": svc.dataset.status(), "sessions": DatasetRecorder.list_sessions()}

    @app.get("/api/dataset/download/{experiment_id}/{session}")
    def dataset_download(experiment_id: str, session: str):
        data = DatasetRecorder.zip_session(experiment_id, session)
        if data is None:
            raise HTTPException(404, "dataset session not found")
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{experiment_id}_{session}.zip"'})

    # -------------------------------------------------------- recordings
    @app.post("/api/tracker/save_recording")
    async def save_recording(req: Request) -> dict[str, Any]:
        rec = ROOT / "recordings"
        rec.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        data, filename = b"", ""
        if "multipart/form-data" in req.headers.get("content-type", ""):
            form = await req.form()
            f = form.get("file")
            if f is not None and hasattr(f, "read"):
                filename = getattr(f, "filename", "") or ""
                data = await f.read()
        if not data:
            data = await req.body()
        name = Path(filename).name if filename else f"PARIKSHAK_{svc.experiment_id}_{ts}.webm"
        if not name.endswith((".webm", ".mp4", ".mkv")):
            name = f"{name}_{ts}.webm"
        p = rec / name
        p.write_bytes(data)
        return {"status": "saved", "filename": name, "filepath": str(p.resolve()),
                "size_mb": round(p.stat().st_size / 1048576, 2), "url": f"/recordings/{name}"}

    @app.get("/api/tracker/recordings")
    def recordings() -> dict[str, Any]:
        rec = ROOT / "recordings"
        files = []
        for f in rec.glob("*"):
            if f.suffix.lower() in (".webm", ".mp4", ".mkv", ".avi") and f.stat().st_size > 1000:
                files.append({"name": f.name, "size_mb": round(f.stat().st_size / 1048576, 2),
                              "mtime": f.stat().st_mtime, "url": f"/recordings/{f.name}"})
        files.sort(key=lambda x: x["mtime"], reverse=True)
        return {"recordings": files, "count": len(files), "directory": str(rec.resolve())}

    # ----------------------------------------------------------- reports
    @app.post("/api/tracker/generate_report")
    def generate_report() -> dict[str, Any]:
        from parikshak.perception.report_generator import generate_structured_text_report, save_reports_to_disk

        res = save_reports_to_disk(svc)
        res["preview_text"] = generate_structured_text_report(svc)
        return res

    @app.get("/api/tracker/download_report/{kind}")
    def download_report(kind: str):
        from parikshak.perception.report_generator import generate_pdf_report, generate_structured_text_report

        ts = time.strftime("%Y%m%d_%H%M%S")
        if kind == "pdf":
            return Response(generate_pdf_report(svc), media_type="application/pdf",
                            headers={"Content-Disposition": f'attachment; filename="PARIKSHAK_REPORT_{svc.experiment_id}_{ts}.pdf"'})
        return Response(generate_structured_text_report(svc).encode("utf-8"), media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="PARIKSHAK_REPORT_{svc.experiment_id}_{ts}.txt"'})

    @app.get("/api/tracker/reports")
    def reports() -> dict[str, Any]:
        rd = ROOT / "reports"
        files = [{"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1), "is_pdf": f.suffix == ".pdf",
                  "mtime": f.stat().st_mtime, "url": f"/reports/{f.name}"}
                 for f in rd.glob("*") if f.suffix in (".txt", ".pdf")]
        files.sort(key=lambda x: x["mtime"], reverse=True)
        return {"reports": files, "count": len(files)}

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Start the PARIKSHAK mission server.")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", help="Do not open a browser tab automatically")
    args = ap.parse_args(argv)

    import uvicorn

    url = f"http://{args.host}:{args.port}/"
    print("\n=======================================================")
    print(f"[OK] PARIKSHAK Mission Server  {url}")
    print(f"   * Landing page : {url}")
    print(f"   * Console      : {url}console")
    print(f"   * Rack tags    : {url}tags")
    print(f"   * SIH deck     : {url}presentation")
    print("=======================================================\n", flush=True)
    if not args.no_browser:
        threading.Timer(2.5, webbrowser.open, args=(url + "console",)).start()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
