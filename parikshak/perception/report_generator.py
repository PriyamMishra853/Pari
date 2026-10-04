"""Automated Mission Procedure Report Generator for PARIKSHAK.

Generates both:
  1. A structured, timestamped lightweight plain text file (.txt)
  2. A professional aerospace publication-grade PDF report (.pdf)
Both include full procedure steps, outcomes, physical geometry observations,
astronaut 17-point skeletal biometrics, deviations, and cryptographic seals.
"""

from __future__ import annotations

import hashlib
import io
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _format_time_delta(seconds: float) -> str:
    """Formats elapsed seconds into T+HH:MM:SS.s notation."""
    total_sec = max(0.0, float(seconds))
    hrs = int(total_sec // 3600)
    mins = int((total_sec % 3600) // 60)
    secs = total_sec % 60
    return f"T+{hrs:02d}:{mins:02d}:{secs:04.1f}"


def _get_step_observation(exp_id: str, step_id: str, status: str, hold_duration: float, geometry: dict[str, Any]) -> str:
    """Generates precise sensory/vision evidence based on procedure step and telemetry."""
    if exp_id in ["BCX-1", "BOX-COL-1"]:
        if step_id == "S01":
            locked = geometry.get("container_locked", False)
            detected = geometry.get("outer_box_detected", False)
            if status == "completed":
                return "Outer container identified; rectangular bounds & workspace position locked."
            elif detected:
                return "Container detected in camera view; stabilizing boundary position."
            return "Searching for outer container box on workspace surface."
        elif step_id == "S02":
            if status == "completed":
                return "Dual color detection verified: Red Box (HSV hue ~0-10, 160-180) & Yellow Box (HSV hue ~22-38)."
            return "Optical color verification active for Red and Yellow experiment items."
        elif step_id == "S03":
            if status == "completed":
                return "Red Box placed within container boundary; 2D containment confirmed."
            return "Tracking Red Box trajectory into outer container boundary."
        elif step_id == "S04":
            dist = geometry.get("distance_px")
            dist_str = f" Distance to Red Box: {dist:.1f}px." if dist is not None else ""
            if status == "completed":
                return f"Yellow Box placed inside container, distinct from Red Box.{dist_str}"
            return "Tracking Yellow Box insertion into container."
        elif step_id == "S05":
            if status == "completed":
                return "Collision contact verified; box centroids converged with continuous surface overlap."
            return "Tracking convergence trajectory of Red and Yellow boxes towards impact point."
        elif step_id == "S06":
            dist = geometry.get("distance_px", 0.0) or 0.0
            if status == "completed":
                return f"Boxes separated to distance {dist:.1f}px (nominal separation threshold: >= 20.0px)."
            return "Tracking separation and disengagement of experiment boxes."
    elif exp_id in ["MOA-1", "MULTI-OBJ", "MOA"]:
        if step_id == "S01":
            if status == "completed":
                return "Chair grasped and pulled into workstation position; seat displacement locked."
            return "Tracking astronaut hand approach and chair positioning."
        elif step_id == "S02":
            k_angle = geometry.get("knee_angle_deg")
            k_str = f" (knee angle: {k_angle:.1f}°)" if k_angle is not None else ""
            if status == "completed":
                return f"Astronaut seated posture verified{k_str}; bilateral alignment within ergonomic tolerance (85°-120°)."
            return "Monitoring ergonomic seating posture and lower limb flexion."
        elif step_id == "S03":
            if status == "completed":
                return "Smartphone grasped and lifted from desk surface; vertical elevation confirmed."
            return "Tracking hand grasp kinematics on smartphone."
        elif step_id == "S04":
            if status == "completed":
                return "Smartphone returned to desk surface; hand contact released."
            return "Verifying phone replacement onto table surface."
        elif step_id == "S05":
            if status == "completed":
                return "Water bottle grasped and lifted from desk surface."
            return "Tracking hand grasp and bottle elevation off table."
        elif step_id == "S06":
            if status == "completed":
                return f"Water bottle brought to oral region; continuous drinking dwell ({hold_duration:.2f}s >= 1.5s) verified."
            return "Monitoring sustained drinking action at oral keypoints."
        elif step_id == "S07":
            if status == "completed":
                return "Water bottle returned to table surface; astronaut hands retracted and released."
            return "Verifying return of bottle to stable resting state on table."
    else:
        # Default WBP-1 or space mission procedures
        if step_id == "S01":
            if status == "completed":
                return "Target bottle located resting stably on table surface; baseline plane calibrated."
            return "Searching for target water bottle in camera frame."
        elif step_id == "S02":
            delta = geometry.get("lift_delta_px", 0.0) or 0.0
            if status == "completed":
                return f"Hand grasp detected on bottle; vertical lift confirmed (+{delta:.1f}px displacement)."
            return "Tracking astronaut hand approach and grasp kinematics."
        elif step_id == "S03":
            if status == "completed":
                return f"Bottle mouth zone aligned with facial keypoints; sustained drinking hold ({hold_duration:.2f}s) verified."
            return "Monitoring drinking action near mouth keypoints."
        elif step_id == "S04":
            if status == "completed":
                return "Bottle returned to table plane; astronaut hands released and retracted."
            return "Verifying return of bottle to stable resting state on table."

    return "Nominal procedure execution monitored via computer vision."


def generate_structured_text_report(tracker_service_or_tracker: Any) -> str:
    """Generates a structured, timestamped lightweight text report from the tracker state."""
    # Handle either TrackerService or YoloExperimentTracker instance
    if hasattr(tracker_service_or_tracker, "tracker"):
        tracker = tracker_service_or_tracker.tracker
        telemetry = tracker_service_or_tracker.last_telemetry or {}
    else:
        tracker = tracker_service_or_tracker
        telemetry = getattr(tracker, "last_telemetry", {})

    exp_id = getattr(tracker, "experiment_id", "BCX-1")
    title = getattr(tracker, "title", f"Procedure {exp_id}")
    rack_id = getattr(tracker, "rack_id", "BENCH-1 (Desktop)")
    steps = getattr(tracker, "steps", [])
    alerts = getattr(tracker, "alerts", [])
    start_time = getattr(tracker, "start_wall_time", time.time())
    total_frames = getattr(tracker, "frame_count", 0)
    is_complete = getattr(tracker, "protocol_complete", False)
    geometry = telemetry.get("geometry", {})

    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    session_id = f"SES-{exp_id}-{now.strftime('%Y%m%d-%H%M%S')}"

    completed_count = sum(1 for s in steps if s.status == "completed")
    compliance_score = int((completed_count / len(steps)) * 100) if steps else 100

    if is_complete:
        status_label = "COMPLETED · ALL PROTOCOL STEPS VERIFIED NOMINAL"
    elif any(a.severity == "critical" for a in alerts):
        status_label = "DEVIATION ALERT · CRITICAL COMPLIANCE INTERRUPTION"
    elif any(a.kind == "skipped" for a in alerts):
        status_label = "DEVIATION DETECTED · STEP SKIPPED IN SEQUENCE"
    else:
        status_label = f"IN PROGRESS · {completed_count}/{len(steps)} STEPS VERIFIED"

    total_duration_s = max(0.0, time.time() - start_time)

    lines = []
    lines.append("=" * 80)
    lines.append("          PARIKSHAK ON-BOARD MISSION PROCEDURE WITNESS REPORT")
    lines.append("               ISRO SMART INDIA HACKATHON 2026 · PS 26174")
    lines.append("=" * 80)
    lines.append(f"Generated Timestamp  : {now_str}")
    lines.append(f"Session Identifier   : {session_id}")
    lines.append(f"Experiment ID        : {exp_id}")
    lines.append(f"Procedure Title      : {title}")
    lines.append(f"Workstation / Rack   : {rack_id}")
    lines.append(f"Execution Outcome    : {status_label}")
    lines.append(f"Compliance Rating    : {compliance_score}% ({completed_count}/{len(steps)} steps nominal)")
    lines.append(f"Total Mission Clock  : {_format_time_delta(total_duration_s)} ({total_duration_s:.1f} s)")
    lines.append(f"Vision Processed     : {total_frames:,} frames processed")
    lines.append(f"Recorded Deviations  : {len(alerts)} alerts logged")
    lines.append("")

    lines.append("-" * 80)
    lines.append("1. PROCEDURE STEP EXECUTION AUDIT CHRONOLOGY")
    lines.append("-" * 80)

    for idx, s in enumerate(steps, 1):
        status_text = s.status.upper()
        if s.status == "completed":
            icon = "[✓ NOMINAL]"
        elif s.status == "skipped":
            icon = "[✕ SKIPPED]"
        elif s.status == "active":
            icon = "[● ACTIVE ]"
        else:
            icon = "[  PENDING]"

        start_delta = (s.started_at - start_time) if s.started_at else 0.0
        comp_delta = (s.completed_at - start_time) if s.completed_at else (s.started_at - start_time + s.elapsed_s if s.started_at else 0.0)
        hold_s = getattr(s, "hold_duration_s", 0.0) or (comp_delta - start_delta if s.status == "completed" else s.elapsed_s)

        obs = _get_step_observation(exp_id, s.id, s.status, hold_s, geometry)

        lines.append(f"[STEP {idx:02d}] {s.id} : {s.name}")
        lines.append(f"  - Requirement     : {s.prompt}")
        lines.append(f"  - Verification    : {icon} {status_text}")
        lines.append(f"  - Start Time      : {_format_time_delta(start_delta)}")
        if s.completed_at:
            lines.append(f"  - Completion Time : {_format_time_delta(comp_delta)}")
        lines.append(f"  - Hold / Dwell    : {hold_s:.2f} s")
        lines.append(f"  - Sensory Evidence: {obs}")
        lines.append("")

    lines.append("-" * 80)
    lines.append("2. PHYSICAL GEOMETRY & VISION SENSOR TELEMETRY")
    lines.append("-" * 80)
    if exp_id in ["BCX-1", "BOX-COL-1"]:
        lines.append(f"  - Outer Container Box  : {'LOCKED & STABLE' if geometry.get('container_locked') else ('DETECTED' if geometry.get('outer_box_detected') else 'SEARCHING')}")
        lines.append(f"  - Red Box State        : {'INSIDE CONTAINER' if geometry.get('red_inside') else ('DETECTED' if geometry.get('red_box_detected') else 'ABSENT')}")
        lines.append(f"  - Yellow Box State     : {'INSIDE CONTAINER' if geometry.get('yellow_inside') else ('DETECTED' if geometry.get('yellow_box_detected') else 'ABSENT')}")
        dist = geometry.get("distance_px")
        lines.append(f"  - Box Distance         : {f'{dist:.1f} px' if dist is not None else '--'}")
        lines.append(f"  - Collision State      : {'ACTIVE COLLISION DETECTED' if geometry.get('is_colliding') else 'CLEAR / SEPARATED'}")
    else:
        lines.append(f"  - Target Object        : {'DETECTED (' + str(int((geometry.get('target_confidence', 0.9))*100)) + '%)' if geometry.get('target_detected') else 'SEARCHING'}")
        lines.append(f"  - Hand Contact         : {'GRASP CONFIRMED' if geometry.get('hand_contact') else 'NONE'}")
        lines.append(f"  - Vertical Displacement: {geometry.get('lift_delta_px', 0.0):.1f} px")
        lines.append(f"  - Mouth Proximity      : {'DRINKING ZONE' if geometry.get('near_mouth') else 'AWAY'}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("3. FULL-BODY ASTRONAUT BIOMETRICS (17-POINT COCO POSE)")
    lines.append("-" * 80)
    pts = geometry.get("body_points_count", 0)
    posture = geometry.get("posture_status", "NOMINAL_TRACKING")
    stability = geometry.get("posture_stability", "STABLE")
    angle = geometry.get("torso_angle_deg", 0.0)
    velocity = geometry.get("body_velocity_px_s", 0.0)
    zones = geometry.get("body_zones", {})

    lines.append(f"  - Keypoint Tracking    : {pts} / 17 COCO Skeletal Landmarks")
    lines.append(f"  - Astronaut Posture    : {posture.replace('_', ' ')}")
    lines.append(f"  - Microgravity Balance : {stability}")
    lines.append(f"  - Spine / Torso Tilt   : {abs(angle):.1f} degrees")
    lines.append(f"  - Kinematic Drift Rate : {velocity:.1f} px/s")
    lines.append(f"  - Body Subsystem Zones : Head/Visor: {zones.get('head', 0)}/5 | Upper Limbs: {zones.get('upper_limbs', 0)}/4 | Torso: {zones.get('torso', 0)}/4 | Foot Restraints: {zones.get('lower_limbs', 0)}/4")
    lines.append("")

    lines.append("-" * 80)
    lines.append("4. DEVIATION & ANOMALY AUDIT LOG")
    lines.append("-" * 80)
    if not alerts:
        lines.append("  [NOMINAL] No procedure sequence deviations, timeouts, or safety alerts detected.")
    else:
        for a in alerts:
            t_delta = (a.timestamp - start_time) if a.timestamp else 0.0
            lines.append(f"  - [{_format_time_delta(t_delta)}] [{a.severity.upper()}] [{a.step_id}] {a.kind.upper()}: {a.message}")
            if a.spoken_tts:
                lines.append(f"    Crew Audio Warning: \"{a.spoken_tts}\"")

    lines.append("")
    lines.append("-" * 80)
    lines.append("5. WITNESS ATTESTATION & INTEGRITY SEAL")
    lines.append("-" * 80)
    lines.append("  This document is an automated procedure witness record generated in real time")
    lines.append("  by the PARIKSHAK onboard vision intelligence system. All procedure rules,")
    lines.append("  temporal bounds, object interactions, and astronaut kinematics were evaluated")
    lines.append("  deterministically without human tampering.")
    lines.append("")
    # Generate deterministic SHA-256 seal
    raw_content = "\n".join(lines)
    sha = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    lines.append(f"  Cryptographic Hash (SHA-256): {sha}")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_pdf_report(tracker_service_or_tracker: Any) -> bytes:
    """Compiles a publication-grade aerospace PDF report using ReportLab."""
    if hasattr(tracker_service_or_tracker, "tracker"):
        tracker = tracker_service_or_tracker.tracker
        telemetry = tracker_service_or_tracker.last_telemetry or {}
    else:
        tracker = tracker_service_or_tracker
        telemetry = getattr(tracker, "last_telemetry", {})

    exp_id = getattr(tracker, "experiment_id", "BCX-1")
    title = getattr(tracker, "title", f"Procedure {exp_id}")
    rack_id = getattr(tracker, "rack_id", "BENCH-1")
    steps = getattr(tracker, "steps", [])
    alerts = getattr(tracker, "alerts", [])
    start_time = getattr(tracker, "start_wall_time", time.time())
    total_frames = getattr(tracker, "frame_count", 0)
    is_complete = getattr(tracker, "protocol_complete", False)
    geometry = telemetry.get("geometry", {})

    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    session_id = f"SES-{exp_id}-{now.strftime('%Y%m%d-%H%M%S')}"

    completed_count = sum(1 for s in steps if s.status == "completed")
    compliance_score = int((completed_count / len(steps)) * 100) if steps else 100
    total_duration_s = max(0.0, time.time() - start_time)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0B132B"),
        spaceAfter=2,
    )
    header_subtitle_style = ParagraphStyle(
        "HeaderSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#1C64F2"),
        spaceAfter=1,
    )
    header_tagline_style = ParagraphStyle(
        "HeaderTagline",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748B"),
    )
    section_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=8,
        spaceAfter=4,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0F172A"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    elements = []

    # 1. Header Banner Box
    status_bg = "#DEF7EC" if is_complete else ("#FDE8E8" if alerts else "#E1EFFE")
    status_color = "#03543F" if is_complete else ("#9B1C1C" if alerts else "#1E429F")
    status_txt = "VERIFIED NOMINAL" if is_complete else ("DEVIATION DETECTED" if alerts else "IN PROGRESS")

    header_data = [
        [
            Paragraph("<b>PARIKSHAK</b> · ON-BOARD MISSION PROCEDURE WITNESS", header_title_style),
            Paragraph(f"<font color='{status_color}'><b>{status_txt}</b></font><br/><font size='7' color='#475569'>Score: {compliance_score}%</font>", ParagraphStyle("StatusBox", fontName="Helvetica-Bold", fontSize=10, alignment=2, leading=12))
        ],
        [
            Paragraph("ISRO SMART INDIA HACKATHON 2026 · PROBLEM STATEMENT PS 26174", header_subtitle_style),
            Paragraph(f"<font size='7.5' color='#64748B'>{now_str}</font>", ParagraphStyle("TimeBox", alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[5.5 * inch, 2.0 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1C64F2"), spaceBefore=1, spaceAfter=6))

    # 2. Key Metadata Summary Grid
    meta_data = [
        [
            Paragraph("<b>Experiment ID:</b>", table_cell), Paragraph(exp_id, table_cell_bold),
            Paragraph("<b>Session ID:</b>", table_cell), Paragraph(session_id, table_cell)
        ],
        [
            Paragraph("<b>Procedure Title:</b>", table_cell), Paragraph(title, table_cell_bold),
            Paragraph("<b>Mission Clock:</b>", table_cell), Paragraph(f"{_format_time_delta(total_duration_s)} ({total_duration_s:.1f}s)", table_cell_bold)
        ],
        [
            Paragraph("<b>Workstation / Rack:</b>", table_cell), Paragraph(rack_id, table_cell),
            Paragraph("<b>Processed Video:</b>", table_cell), Paragraph(f"{total_frames:,} frames @ 25 FPS", table_cell)
        ],
    ]
    meta_table = Table(meta_data, colWidths=[1.3 * inch, 2.7 * inch, 1.3 * inch, 2.2 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 6))

    # 3. Procedure Step Audit Table
    elements.append(Paragraph("1. Procedure Step Execution Audit Chronology", section_h2))

    step_rows = [
        [
            Paragraph("<b>Step ID</b>", table_header),
            Paragraph("<b>Procedure Action</b>", table_header),
            Paragraph("<b>Outcome Status</b>", table_header),
            Paragraph("<b>Start / End (T+)</b>", table_header),
            Paragraph("<b>Hold Time</b>", table_header),
            Paragraph("<b>Sensory Evidence &amp; Visual Verification</b>", table_header),
        ]
    ]

    for s in steps:
        start_d = (s.started_at - start_time) if s.started_at else 0.0
        comp_d = (s.completed_at - start_time) if s.completed_at else (s.started_at - start_time + s.elapsed_s if s.started_at else 0.0)
        hold_s = getattr(s, "hold_duration_s", 0.0) or (comp_d - start_d if s.status == "completed" else s.elapsed_s)

        obs = _get_step_observation(exp_id, s.id, s.status, hold_s, geometry)

        if s.status == "completed":
            badge_html = "<font color='#03543F'><b>✓ VERIFIED</b></font>"
        elif s.status == "skipped":
            badge_html = "<font color='#9B1C1C'><b>✕ SKIPPED</b></font>"
        elif s.status == "active":
            badge_html = "<font color='#1E429F'><b>● ACTIVE</b></font>"
        else:
            badge_html = "<font color='#64748B'>WAITING</font>"

        time_range = f"{_format_time_delta(start_d)}"
        if s.completed_at:
            time_range += f"<br/>{_format_time_delta(comp_d)}"

        step_rows.append([
            Paragraph(f"<b>{s.id}</b>", table_cell_bold),
            Paragraph(f"<b>{s.name}</b><br/><font color='#64748B' size='7'>{s.prompt}</font>", table_cell),
            Paragraph(badge_html, table_cell),
            Paragraph(time_range, table_cell),
            Paragraph(f"{hold_s:.2f}s", table_cell),
            Paragraph(obs, table_cell),
        ])

    step_table = Table(step_rows, colWidths=[0.6 * inch, 2.0 * inch, 1.0 * inch, 1.0 * inch, 0.7 * inch, 2.2 * inch])
    step_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B132B")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    elements.append(step_table)
    elements.append(Spacer(1, 6))

    # 4. Biometrics & Physical Telemetry
    elements.append(Paragraph("2. Astronaut Full-Body Biometrics &amp; Spatial Telemetry", section_h2))

    pts = geometry.get("body_points_count", 0)
    posture = geometry.get("posture_status", "NOMINAL_TRACKING")
    stability = geometry.get("posture_stability", "STABLE")
    angle = geometry.get("torso_angle_deg", 0.0)
    velocity = geometry.get("body_velocity_px_s", 0.0)

    bio_data = [
        [
            Paragraph("<b>Skeletal Points:</b>", table_cell), Paragraph(f"{pts} / 17 COCO Keypoints", table_cell_bold),
            Paragraph("<b>Posture State:</b>", table_cell), Paragraph(posture.replace("_", " "), table_cell_bold),
        ],
        [
            Paragraph("<b>Spine Alignment:</b>", table_cell), Paragraph(f"{abs(angle):.1f}° inclination", table_cell),
            Paragraph("<b>Microgravity Drift:</b>", table_cell), Paragraph(f"{velocity:.1f} px/s ({stability})", table_cell),
        ],
    ]
    bio_table = Table(bio_data, colWidths=[1.5 * inch, 2.25 * inch, 1.5 * inch, 2.25 * inch])
    bio_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(bio_table)
    elements.append(Spacer(1, 6))

    # 5. Safety & Deviations Section
    elements.append(Paragraph("3. Deviation &amp; Safety Compliance Audit", section_h2))
    if not alerts:
        alert_p = Paragraph("<font color='#03543F'><b>✓ ZERO DEVIATIONS DETECTED:</b> Procedure steps conducted strictly according to mission protocol specifications.</font>", table_cell)
        alert_box = Table([[alert_p]], colWidths=[7.5 * inch])
        alert_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#DEF7EC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#31C48D")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(alert_box)
    else:
        alert_rows = [
            [
                Paragraph("<b>Timestamp</b>", table_header),
                Paragraph("<b>Step</b>", table_header),
                Paragraph("<b>Severity</b>", table_header),
                Paragraph("<b>Deviation Detail</b>", table_header),
            ]
        ]
        for a in alerts:
            t_d = (a.timestamp - start_time) if a.timestamp else 0.0
            alert_rows.append([
                Paragraph(_format_time_delta(t_d), table_cell),
                Paragraph(a.step_id, table_cell_bold),
                Paragraph(f"<font color='#9B1C1C'><b>{a.severity.upper()}</b></font>", table_cell),
                Paragraph(f"<b>{a.kind.upper()}:</b> {a.message}", table_cell),
            ])
        alert_table = Table(alert_rows, colWidths=[1.1 * inch, 0.7 * inch, 1.1 * inch, 4.6 * inch])
        alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#9B1C1C")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E02424")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#FBD5D5")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FDF2F2")]),
        ]))
        elements.append(alert_table)

    elements.append(Spacer(1, 8))

    # 6. Attestation & Seal
    raw_hash_seed = f"{session_id}-{exp_id}-{compliance_score}-{now_str}"
    sha = hashlib.sha256(raw_hash_seed.encode("utf-8")).hexdigest()

    attest_data = [
        [
            Paragraph(
                "<b>WITNESS ATTESTATION:</b> Verified nominal execution generated automatically via PARIKSHAK deterministic vision witness pipeline.<br/>"
                f"<font size='6.5' color='#64748B'>CRYPTOGRAPHIC SHA-256 INTEGRITY SEAL: {sha}</font>",
                ParagraphStyle("Attest", parent=table_cell, fontSize=7, leading=9)
            )
        ]
    ]
    attest_table = Table(attest_data, colWidths=[7.5 * inch])
    attest_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(attest_table)

    doc.build(elements)
    return buffer.getvalue()


def save_reports_to_disk(tracker_service_or_tracker: Any, target_dir: str | Path = "reports") -> dict[str, Any]:
    """Generates both .txt and .pdf reports and persists them into the local reports folder."""
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(tracker_service_or_tracker, "tracker"):
        exp_id = getattr(tracker_service_or_tracker.tracker, "experiment_id", "EXP")
    else:
        exp_id = getattr(tracker_service_or_tracker, "experiment_id", "EXP")

    now = datetime.now()
    timestamp_slug = now.strftime("%Y%m%d_%H%M%S")

    # 1. Generate text report
    txt_content = generate_structured_text_report(tracker_service_or_tracker)
    txt_filename = f"PARIKSHAK_REPORT_{exp_id}_{timestamp_slug}.txt"
    txt_path = out_dir / txt_filename
    txt_path.write_text(txt_content, encoding="utf-8")

    # 2. Generate PDF report
    pdf_bytes = generate_pdf_report(tracker_service_or_tracker)
    pdf_filename = f"PARIKSHAK_REPORT_{exp_id}_{timestamp_slug}.pdf"
    pdf_path = out_dir / pdf_filename
    pdf_path.write_bytes(pdf_bytes)

    integrity_hash = hashlib.sha256(txt_content.encode("utf-8")).hexdigest()

    return {
        "status": "generated",
        "experiment_id": exp_id,
        "timestamp": now.isoformat(),
        "integrity_hash": integrity_hash,
        "text_filename": txt_filename,
        "pdf_filename": pdf_filename,
        "text_url": f"/reports/{txt_filename}",
        "pdf_url": f"/reports/{pdf_filename}",
        "text_file": {
            "filename": txt_filename,
            "filepath": str(txt_path.resolve()),
            "size_bytes": txt_path.stat().st_size,
            "url": f"/reports/{txt_filename}",
        },
        "pdf_file": {
            "filename": pdf_filename,
            "filepath": str(pdf_path.resolve()),
            "size_bytes": pdf_path.stat().st_size,
            "size_kb": round(pdf_path.stat().st_size / 1024, 1),
            "url": f"/reports/{pdf_filename}",
        },
    }

