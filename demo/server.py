"""A local web server for the demo.

    python -m demo [--port 8765] [--no-browser]

Runs offline: every scenario is replayed by the engine on this machine the first
time it is opened. The page is the same file `tools/make_demo_page.py` embeds
the replays into for sharing - with no embedded data it asks this server.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any
import numpy as np
from starlette.requests import Request

from demo import guided
from demo.scenarios import DEFAULT, GROUP_ORDER, catalogue, replay_json, results

STATIC = Path(__file__).with_name("static")
PAGE = STATIC / "index.html"

#: The page is authored without <html>/<head>/<body> so the same file can be
#: published as it is; served locally it is wrapped here.
_SHELL_HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width, initial-scale=1, '
               'viewport-fit=cover"></head><body>')
_SHELL_TAIL = "</body></html>"


def page(bundle: dict[str, Any] | None = None, *, standalone: bool = True) -> str:
    """The demo page, optionally with every replay embedded.

    `standalone` wraps it in a full HTML document, for serving or opening from
    disk; without it the bare page is returned, ready to publish.
    """
    body = PAGE.read_text(encoding="utf-8")
    if bundle is not None:
        data = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
        body = body.replace("<!--BUNDLE-->", f"<script>window.PARIKSHAK_BUNDLE={data};</script>")
    return f"{_SHELL_HEAD}{body}{_SHELL_TAIL}" if standalone else body


def create_app():
    from fastapi import FastAPI, HTTPException, UploadFile, File, Request
    from fastapi.responses import HTMLResponse, Response
    from parikshak.perception.tracker_service import get_tracker_service
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from pathlib import Path

    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="PARIKSHAK demo", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    tracker_svc = get_tracker_service()

    static_p = Path("demo/static")
    if static_p.exists():
        app.mount("/static", StaticFiles(directory=str(static_p)), name="static")

    recordings_p = Path("recordings")
    recordings_p.mkdir(exist_ok=True)
    app.mount("/recordings", StaticFiles(directory=str(recordings_p)), name="recordings")

    reports_p = Path("reports")
    reports_p.mkdir(exist_ok=True)
    app.mount("/reports", StaticFiles(directory=str(reports_p)), name="reports")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        landing_p = Path("frontend/index.html")
        if landing_p.exists():
            return landing_p.read_text(encoding="utf-8")
        return page()

    @app.get("/console", response_class=HTMLResponse)
    @app.get("/app", response_class=HTMLResponse)
    @app.get("/workspace", response_class=HTMLResponse)
    def console_page() -> str:
        console_p = Path("frontend/console.html")
        if console_p.exists():
            return console_p.read_text(encoding="utf-8")
        return page()

    @app.get("/presentation", response_class=HTMLResponse)
    def presentation_page():
        p = Path("demo/static/sih_presentation.html")
        if p.exists():
            return HTMLResponse(p.read_text(encoding="utf-8"))
        raise HTTPException(404, "Presentation page not found")

    @app.get("/api/download_pptx")
    def download_pptx():
        p = Path("PARIKSHAK_SIH2026_Submission.pptx")
        if p.exists():
            return FileResponse(
                path=str(p),
                filename="PARIKSHAK_SIH2026_Submission.pptx",
                media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
        raise HTTPException(404, "PowerPoint deck not found")

    @app.get("/api/scenarios")
    def scenarios() -> dict[str, Any]:
        return {"default": DEFAULT, "groups": list(GROUP_ORDER), "scenarios": catalogue()}

    @app.get("/api/scenario/{scenario_id}")
    def scenario(scenario_id: str) -> Response:
        try:
            return Response(replay_json(scenario_id), media_type="application/json")
        except KeyError:
            raise HTTPException(404, f"no scenario {scenario_id!r}") from None

    @app.get("/api/results")
    def measured() -> dict[str, Any]:
        return results()

    # -- guided runs: the engine walking someone through an experiment ----
    @app.get("/api/guided/experiments")
    def guided_experiments() -> dict[str, Any]:
        return {"experiments": guided.experiments()}

    @app.post("/api/guided/start")
    def guided_start(payload: dict) -> dict[str, Any]:
        try:
            session_id, run = guided.start(str(payload.get("experiment", "")))
        except KeyError as exc:
            raise HTTPException(404, f"no experiment {exc}") from None
        return {"session": session_id, "static": run.static(), "state": run.view()}

    @app.post("/api/guided/act")
    def guided_act(payload: dict) -> dict[str, Any]:
        try:
            run = guided.get(str(payload.get("session", "")))
        except KeyError:
            raise HTTPException(410, "this run is no longer open - start a new one") from None
        try:
            run.perform(str(payload.get("action", "")))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"state": run.view()}

    # -- Live YOLO and OpenCV Human Activity Tracker endpoints -----------
    @app.get("/api/tracker/experiments")
    def tracker_experiments() -> dict[str, Any]:
        return {
            "experiments": [
                {
                    "id": "WBP-1",
                    "title": "Water Bottle Protocol (Activity Benchmark)",
                    "category": "Interactive Benchmark & Prototype",
                    "rack": "BENCH-1 (Desktop / Tabletop)",
                    "steps_count": 4,
                    "target_object": "Water Bottle",
                    "description": "Validates 4 steps: bottle identified -> hand grasps and lifts -> drinks water at mouth (held >= 1.5s) -> bottle returned to table surface and released.",
                },
                {
                    "id": "MOA-1",
                    "title": "Multi Object Experiment (Chair, Phone & Bottle)",
                    "category": "Multi-Object HAR & Ergonomic Posture",
                    "rack": "BENCH-1 (Desktop Workspace / Ergonomics Lab)",
                    "steps_count": 7,
                    "target_object": "Chair, Smartphone & Water Bottle",
                    "description": "Validates 7 activities with live camera verification: S01 Pull Chair -> S02 Sit Down on Chair (knee angle 85°-120°) -> S03 Pick Up Smartphone -> S04 Return Smartphone to Desk -> S05 Grasp and Lift Bottle -> S06 Drink Water (held >= 1.5s) -> S07 Return Bottle & Release Hands.",
                },
                {
                    "id": "BCX-1",
                    "title": "BCX-1: Two-Box Collision in Container (Red & Yellow)",
                    "category": "Physical Dynamics & Color HAR",
                    "rack": "BENCH-1 (Desktop Workspace / Glovebox)",
                    "steps_count": 6,
                    "target_object": "Red & Yellow Boxes",
                    "description": "Validates real-time color tracking & collision dynamics: S01 Identify container -> S02 Verify box colors (Red & Yellow) -> S03 Place Red box -> S04 Place Yellow box -> S05 Collide boxes -> S06 Separate boxes. Guards against skipped placement, uncalibrated colors, and out-of-bounds collision.",
                },
                {
                    "id": "CRX-2",
                    "title": "CRX-2 : Colloid Resuspension and Cold Return",
                    "category": "ISRO Space Experiment (Microgravity)",
                    "rack": "MSG-A (Microgravity Science Glovebox)",
                    "steps_count": 8,
                    "target_object": "Colloid Vial B",
                    "description": "Full space flight procedure: foot restraint lock -> cold locker retrieval -> processing unit check -> 10x resuspension agitation -> tray settling -> restowage.",
                },
                {
                    "id": "CSP-1",
                    "title": "CSP-1 : Colloid Sample Processing",
                    "category": "ISRO Space Experiment (Microgravity)",
                    "rack": "MSG-A (Microgravity Science Glovebox)",
                    "steps_count": 14,
                    "target_object": "Sample Cartridge & Vial",
                    "description": "Complete 14-step microgravity colloid sample processing protocol with glovebox latch and cartridge lock verification.",
                },
            ]
        }

    @app.post("/api/tracker/set_experiment")
    def tracker_set_experiment(payload: dict) -> dict[str, Any]:
        exp_id = str(payload.get("experiment_id", "WBP-1"))
        return tracker_svc.set_experiment(exp_id)

    @app.post("/api/tracker/reset")
    def tracker_reset() -> dict[str, Any]:
        return tracker_svc.reset()

    @app.post("/api/tracker/frame")
    def tracker_process_frame(payload: dict) -> dict[str, Any]:
        frame_data = str(payload.get("frame", ""))
        if not frame_data:
            raise HTTPException(400, "Missing frame data")
        res = tracker_svc.process_b64_frame(frame_data)
        if "error" in res:
            raise HTTPException(400, res["error"])
        return res

    @app.get("/api/tracker/telemetry")
    def tracker_telemetry() -> dict[str, Any]:
        with tracker_svc.lock:
            if not tracker_svc.last_telemetry:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                _, telem = tracker_svc.tracker.process_frame(dummy)
                tracker_svc.last_telemetry = telem
            return tracker_svc.last_telemetry

    @app.post("/api/tracker/upload_video")
    async def tracker_upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
        upload_path = Path("runs/uploads") / file.filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        upload_path.write_bytes(content)
        return tracker_svc.load_video_file(upload_path)

    @app.post("/api/tracker/load_demo_video")
    def tracker_load_demo_video() -> dict[str, Any]:
        demo_path = Path("runs/uploads/demo_bottle_run.mp4")
        if not demo_path.exists():
            tracker_svc.generate_demo_video(str(demo_path))
        return tracker_svc.load_video_file(demo_path)

    @app.get("/api/tracker/next_video_frame")
    def tracker_next_video_frame() -> dict[str, Any]:
        res = tracker_svc.get_next_video_frame()
        if "error" in res:
            raise HTTPException(400, res["error"])
        return res

    @app.post("/api/tracker/simulate")
    def tracker_simulate(payload: dict) -> dict[str, Any]:
        event_name = str(payload.get("event", "nominal_step"))
        return tracker_svc.simulate_event(event_name)

    @app.get("/api/tracker/cameras")
    def tracker_cameras() -> dict[str, Any]:
        return {"cameras": tracker_svc.list_available_cameras()}

    @app.post("/api/tracker/start_camera")
    def tracker_start_camera(payload: dict | None = None) -> dict[str, Any]:
        raw_idx = (payload or {}).get("camera_index", -1)
        try:
            idx = int(raw_idx)
        except (ValueError, TypeError):
            idx = -1
        return tracker_svc.start_local_camera(idx)

    @app.post("/api/tracker/stop_camera")
    def tracker_stop_camera() -> dict[str, Any]:
        return tracker_svc.stop_local_camera()

    @app.get("/api/tracker/camera_frame")
    def tracker_camera_frame() -> dict[str, Any]:
        return tracker_svc.get_camera_frame_b64()

    @app.get("/api/tracker/camera_stream")
    def tracker_camera_stream():
        import time
        from fastapi.responses import StreamingResponse

        def stream_generator():
            while True:
                frame_bytes = tracker_svc.get_camera_frame_mjpeg()
                if frame_bytes is None:
                    time.sleep(0.04)
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(0.033)

        return StreamingResponse(stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/tracker/save_recording")
    async def tracker_save_recording(request: Request) -> dict[str, Any]:
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(exist_ok=True)
        import time

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        content_type = request.headers.get("content-type", "")
        file_bytes = b""
        filename = ""

        if "multipart/form-data" in content_type:
            form = await request.form()
            file_item = form.get("file")
            if file_item is not None and hasattr(file_item, "read"):
                filename = getattr(file_item, "filename", "") or ""
                file_bytes = await file_item.read()

        if not file_bytes:
            file_bytes = await request.body()

        if filename:
            safe_name = Path(filename).name
            if not safe_name.endswith((".webm", ".mp4", ".mkv", ".avi")):
                safe_name = f"{safe_name}_{timestamp}.webm"
        else:
            safe_name = f"PARIKSHAK_EXP_{timestamp}.webm"

        target_path = recordings_dir / safe_name
        target_path.write_bytes(file_bytes)
        file_size = target_path.stat().st_size
        return {
            "status": "saved",
            "filename": safe_name,
            "filepath": str(target_path.resolve()),
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "url": f"/recordings/{safe_name}",
        }

    @app.get("/api/tracker/recordings")
    def tracker_list_recordings() -> dict[str, Any]:
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(exist_ok=True)
        files = []
        for ext in ("*.webm", "*.mp4", "*.mkv", "*.avi"):
            for f in recordings_dir.glob(ext):
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "filepath": str(f.resolve()),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "mtime": stat.st_mtime,
                    "url": f"/recordings/{f.name}",
                })
        files.sort(key=lambda x: x["mtime"], reverse=True)
        return {"recordings": files, "count": len(files), "directory": str(recordings_dir.resolve())}

    @app.post("/api/tracker/generate_report")
    def tracker_generate_report() -> dict[str, Any]:
        from parikshak.perception.report_generator import generate_structured_text_report, save_reports_to_disk
        res = save_reports_to_disk(tracker_svc)
        res["preview_text"] = generate_structured_text_report(tracker_svc)
        return res

    @app.get("/api/tracker/download_report/text")
    def tracker_download_text_report():
        from parikshak.perception.report_generator import generate_structured_text_report
        txt = generate_structured_text_report(tracker_svc)
        exp_id = tracker_svc.experiment_id
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"PARIKSHAK_REPORT_{exp_id}_{timestamp}.txt"
        return Response(
            content=txt.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/tracker/download_report/pdf")
    def tracker_download_pdf_report():
        from parikshak.perception.report_generator import generate_pdf_report
        pdf_bytes = generate_pdf_report(tracker_svc)
        exp_id = tracker_svc.experiment_id
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"PARIKSHAK_REPORT_{exp_id}_{timestamp}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/tracker/reports")
    def tracker_list_reports() -> dict[str, Any]:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        files = []
        for ext in ("*.txt", "*.pdf"):
            for f in reports_dir.glob(ext):
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "filepath": str(f.resolve()),
                    "size_bytes": stat.st_size,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "is_pdf": f.suffix.lower() == ".pdf",
                    "mtime": stat.st_mtime,
                    "url": f"/reports/{f.name}",
                })
        files.sort(key=lambda x: x["mtime"], reverse=True)
        return {"reports": files, "count": len(files), "directory": str(reports_dir.resolve())}

    return app




def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Open the PARIKSHAK demo in a browser.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)

    import uvicorn

    url = f"http://127.0.0.1:{args.port}/"
    print(f"PARIKSHAK demo at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0
    app = FastAPI()
