"""PARIKSHAK Backend Application & Flight Telemetry Server.

Hosts:
  1. Mission Landing Page (/) -> frontend/index.html
  2. Interactive Flight Console (/console, /app) -> frontend/console.html
  3. SIH Presentation Deck (/presentation) -> frontend/sih_presentation.html
  4. Real-time Edge AI Tracking & Video APIs (/api/tracker/*)
  5. Procedure Verification & Golden Scenario Replay APIs (/api/scenarios, /api/results)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from starlette.requests import Request

from demo import scenarios as S

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
LANDING_PAGE = FRONTEND_DIR / "index.html"
CONSOLE_PAGE = FRONTEND_DIR / "console.html"


def get_landing_page() -> str:
    """Returns the interactive space-themed landing page."""
    if LANDING_PAGE.exists():
        return LANDING_PAGE.read_text(encoding="utf-8")
    return "<!doctype html><html><head><title>PARIKSHAK</title></head><body><h1>PARIKSHAK</h1><p>Mission Landing Page</p></body></html>"


def get_console_page(bundle: dict[str, Any] | None = None) -> str:
    """Returns the full experiment workstation console page with bundled data."""
    if not CONSOLE_PAGE.exists():
        fallback = ROOT / "demo" / "static" / "index.html"
        body = fallback.read_text(encoding="utf-8") if fallback.exists() else "<h1>Console Not Found</h1>"
    else:
        body = CONSOLE_PAGE.read_text(encoding="utf-8")

    if bundle is not None:
        data = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
        body = body.replace("<!--BUNDLE-->", f"<script>window.PARIKSHAK_BUNDLE={data};</script>")
    return body


def create_app():
    from fastapi import FastAPI, HTTPException, UploadFile, File, Request
    from fastapi.responses import HTMLResponse, Response, FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from parikshak.perception.tracker_service import get_tracker_service
    from backend.flight_copilot import get_flight_copilot

    app = FastAPI(title="PARIKSHAK Mission Server", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    tracker_svc = get_tracker_service()
    copilot = get_flight_copilot()

    # Mount static assets
    static_p = FRONTEND_DIR / "static"
    static_p.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_p)), name="static")

    recordings_p = ROOT / "recordings"
    recordings_p.mkdir(exist_ok=True)
    app.mount("/recordings", StaticFiles(directory=str(recordings_p)), name="recordings")

    reports_p = ROOT / "reports"
    reports_p.mkdir(exist_ok=True)
    app.mount("/reports", StaticFiles(directory=str(reports_p)), name="reports")

    # 1. Main Landing Page
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return get_landing_page()

    # 2. Interactive Mission Console (Workstation)
    @app.get("/console", response_class=HTMLResponse)
    @app.get("/app", response_class=HTMLResponse)
    @app.get("/workspace", response_class=HTMLResponse)
    def console() -> str:
        return get_console_page()

    # 3. SIH Presentation Deck
    @app.get("/presentation", response_class=HTMLResponse)
    def presentation_page():
        p = FRONTEND_DIR / "sih_presentation.html"
        if not p.exists():
            p = ROOT / "demo" / "static" / "sih_presentation.html"
        if p.exists():
            return HTMLResponse(p.read_text(encoding="utf-8"))
        raise HTTPException(404, "Presentation page not found")

    @app.get("/api/download_pptx")
    def download_pptx():
        p = ROOT / "PARIKSHAK_SIH2026_Submission.pptx"
        if p.exists():
            return FileResponse(
                path=str(p),
                filename="PARIKSHAK_SIH2026_Submission.pptx",
                media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
        raise HTTPException(404, "Presentation file not found")

    # 4. Scenario Replay & Verification APIs
    @app.get("/api/scenarios")
    def list_scenarios() -> dict[str, Any]:
        return {
            "default": S.DEFAULT,
            "groups": S.GROUP_ORDER,
            "scenarios": S.catalogue(),
        }

    @app.get("/api/scenario/{scenario_id}")
    def get_scenario(scenario_id: str) -> dict[str, Any]:
        try:
            return S.replay(scenario_id)
        except KeyError:
            raise HTTPException(404, f"unknown scenario {scenario_id!r}")

    @app.get("/api/results")
    def get_results() -> dict[str, Any]:
        eval_path = ROOT / "runs" / "eval.json"
        eval_crx2 = ROOT / "runs" / "eval_crx2.json"
        bench_path = ROOT / "runs" / "bench.json"
        soak_path = ROOT / "runs" / "soak.json"

        return {
            "csp1": json.loads(eval_path.read_text("utf-8")) if eval_path.exists() else None,
            "crx2": json.loads(eval_crx2.read_text("utf-8")) if eval_crx2.exists() else None,
            "bench": json.loads(bench_path.read_text("utf-8")) if bench_path.exists() else None,
            "soak": json.loads(soak_path.read_text("utf-8")) if soak_path.exists() else None,
        }

    # 5. Tracker & Experiment APIs
    @app.get("/api/tracker/experiments")
    def tracker_get_experiments() -> dict[str, Any]:
        return tracker_svc.get_experiments_list()

    @app.post("/api/tracker/experiment/{experiment_id}")
    def tracker_set_experiment(experiment_id: str) -> dict[str, Any]:
        return tracker_svc.set_experiment(experiment_id)

    @app.get("/api/tracker/cameras")
    def tracker_list_cameras() -> dict[str, Any]:
        cams = tracker_svc.list_available_cameras()
        return {"cameras": cams}

    @app.post("/api/tracker/start_camera")
    def tracker_start_camera(cam_idx: int = -1) -> dict[str, Any]:
        return tracker_svc.start_local_camera(cam_idx)

    @app.post("/api/tracker/stop_camera")
    def tracker_stop_camera() -> dict[str, Any]:
        return tracker_svc.stop_local_camera()

    @app.get("/api/tracker/camera_frame")
    def tracker_camera_frame() -> dict[str, Any]:
        return tracker_svc.get_camera_frame_b64()

    @app.get("/api/tracker/stream")
    def tracker_stream():
        def frame_generator():
            while True:
                jpeg_bytes = tracker_svc.get_camera_frame_mjpeg()
                if jpeg_bytes is None:
                    time.sleep(0.04)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
                time.sleep(0.03)

        return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/tracker/frame")
    async def tracker_process_frame(req: Request) -> dict[str, Any]:
        data = await req.json()
        b64_image = data.get("image", "")
        if not b64_image:
            raise HTTPException(400, "image base64 string required")
        return tracker_svc.process_client_frame(b64_image)

    @app.post("/api/tracker/reset")
    @app.post("/api/tracker/restart")
    def tracker_reset() -> dict[str, Any]:
        return tracker_svc.reset()

    @app.get("/api/tracker/telemetry")
    def tracker_telemetry() -> dict[str, Any]:
        return tracker_svc.get_telemetry()

    @app.post("/api/tracker/set_camera_rotation")
    async def tracker_set_rotation(req: Request) -> dict[str, Any]:
        data = await req.json()
        angle = float(data.get("angle", 0.0))
        tracker_svc.set_camera_rotation(angle)
        return tracker_svc.get_rotation_test_telemetry()

    @app.get("/api/tracker/rotation_test")
    def tracker_rotation_test() -> dict[str, Any]:
        return tracker_svc.get_rotation_test_telemetry()

    @app.post("/api/tracker/toggles")
    async def tracker_set_toggles(req: Request) -> dict[str, Any]:
        data = await req.json()
        return {"toggles": tracker_svc.set_toggles(data)}

    @app.get("/api/tracker/sam3d_state")
    def tracker_sam3d_state() -> dict[str, Any]:
        with tracker_svc.lock:
            state_data = tracker_svc.last_telemetry.get("rack_hmr")
            if not state_data:
                dummy_state = tracker_svc.rack_hmr.evaluate_rack_pose({})
                state_data = tracker_svc.rack_hmr.to_dict(dummy_state)
            return state_data

    @app.post("/api/guide/review_step")
    async def guide_review_step(req: Request) -> dict[str, Any]:
        data = await req.json()
        exp_id = data.get("experiment_id", tracker_svc.experiment_id)
        step_id = data.get("step_id", tracker_svc.last_telemetry.get("step_id", "S01"))
        return copilot.review_current_step(exp_id, step_id, tracker_svc.last_telemetry)

    @app.post("/api/guide/convey_next")
    async def guide_convey_next(req: Request) -> dict[str, Any]:
        data = await req.json()
        exp_id = data.get("experiment_id", tracker_svc.experiment_id)
        curr_step = data.get("current_step_id", tracker_svc.last_telemetry.get("step_id", "S01"))
        return copilot.convey_next_step(exp_id, curr_step, tracker_svc.last_telemetry)

    @app.post("/api/experiment/create_custom")
    async def experiment_create_custom(req: Request) -> dict[str, Any]:
        data = await req.json()
        title = data.get("title", "Custom Payload Procedure")
        description = data.get("description", "User-defined on-board experiment")
        steps = data.get("steps", [])
        return copilot.create_custom_experiment(title, description, steps)

    @app.post("/api/tracker/upload_video")
    async def tracker_upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
        upload_dir = ROOT / "runs" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename
        contents = await file.read()
        file_path.write_bytes(contents)
        res = tracker_svc.load_video_file(file_path)
        res["filename"] = file.filename
        return res

    @app.get("/api/tracker/video_frame")
    def tracker_video_frame() -> dict[str, Any]:
        return tracker_svc.get_next_video_frame()

    @app.post("/api/tracker/save_recording")
    async def tracker_save_recording(file: UploadFile = File(...)) -> dict[str, Any]:
        recordings_dir = ROOT / "recordings"
        recordings_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_id = tracker_svc.experiment_id
        safe_name = f"RUN_{exp_id}_{timestamp}.webm"
        target_path = recordings_dir / safe_name
        contents = await file.read()
        target_path.write_bytes(contents)
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
        recordings_dir = ROOT / "recordings"
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
        reports_dir = ROOT / "reports"
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
    ap = argparse.ArgumentParser(description="Start PARIKSHAK Mission Telemetry Server.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", help="Do not open a browser tab automatically")
    args = ap.parse_args(argv)

    import uvicorn

    url = f"http://{args.host}:{args.port}/"
    print(f"\n=======================================================")
    print(f"[OK] PARIKSHAK Mission Server live at {url}")
    print(f"   * Landing Page:   {url}")
    print(f"   * Flight Console: {url}console")
    print(f"   * SIH Deck:       {url}presentation")
    print(f"=======================================================\n")

    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
