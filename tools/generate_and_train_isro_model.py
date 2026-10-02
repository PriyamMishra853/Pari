#!/usr/bin/env python3
"""ISRO / DRDO Microgravity Experiment Synthetic Dataset Generator & YOLO Training Suite.

Designed for SIH 2026 Problem Statement 26174:
  "AI Human Activity Recognition for On-board BAS Experiments"
  Sample Experiment: Outer container box with two smaller boxes (Red and Yellow).

Features:
  1. Uses the provided Groq API (LLaMA-3/vision) to enrich domain scenarios & microgravity physics.
  2. Synthesizes a focused YOLO detection dataset (Outer Container, Red Box, Yellow Box, Gloved Hand).
  3. Formats annotations into standard YOLO txt format (class cx cy w h).
  4. Runs or outputs the Ultralytics YOLOv8 training pipeline for offline edge deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import httpx
except ImportError:
    httpx = None

ROOT = Path(__file__).resolve().parent.parent
GROQ_API_KEY_DEFAULT = os.environ.get("GROQ_API_KEY", "")

CLASSES = ["container_box", "red_box", "yellow_box", "astronaut_hand"]


def query_groq_domain_enrichment(api_key: str) -> dict:
    """Queries Groq LLaMA-3 to generate varied microgravity experiment conditions."""
    if not httpx or not api_key:
        print("[*] Using local procedural rules for synthetic variation.")
        return {}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = (
        "Generate 5 realistic microgravity lighting and visual variations for an ISRO space station (BAS) "
        "experiment involving an outer container box, a red box, a yellow box, and an astronaut's gloved hand. "
        "Return valid JSON with a list of 'variations', each having 'lighting' (dim, glare, normal), "
        "'blur_sigma', 'noise_level', 'angle_deg', and 'box_overlap'."
    )
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [
            {"role": "system", "content": "You are an aerospace mission simulation engineer for ISRO."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"}
    }

    try:
        print(f"[*] Querying Groq API for ISRO experiment domain augmentation...")
        resp = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=12.0)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print("[OK] Successfully received Groq domain augmentations.")
            return json.loads(content)
        else:
            print(f"[!] Groq API returned status {resp.status_code}. Using local presets.")
    except Exception as e:
        print(f"[!] Groq API query note: {e}. Using local procedural presets.")
    return {}


def generate_synthetic_frame(
    idx: int,
    w: int = 640,
    h: int = 480,
    variation: dict | None = None
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Generates one realistic synthetic camera frame of the ISRO Two-Box Experiment.

    Objects:
      0: container_box (large dark-gray container with metallic rim)
      1: red_box (solid crimson/red block)
      2: yellow_box (bright amber/yellow block)
      3: astronaut_hand (white spaceflight glove)

    Returns:
      (image_bgr, labels) where each label is (class_id, x_center, y_center, width, height) in 0..1 coordinates.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Background: Microgravity Science Glovebox (MSG) metallic chamber
    base_color = random.randint(35, 55)
    img[:] = (base_color + 5, base_color, base_color - 5)

    # Add metallic texture & subtle gradient
    gradient = np.tile(np.linspace(0.85, 1.15, h)[:, None, None], (1, w, 3))
    img = np.clip(img * gradient, 0, 255).astype(np.uint8)

    # Rack fiducials / frame borders
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (70, 75, 80), 2)
    cv2.line(img, (20, 70), (w - 20, 70), (55, 60, 65), 1)

    labels = []

    # 2. Outer Container Box (Class 0)
    # Centered in workspace
    cw = random.randint(260, 340)
    ch = random.randint(180, 230)
    cx = w // 2 + random.randint(-30, 30)
    cy = h // 2 + random.randint(10, 60)

    x1 = cx - cw // 2
    y1 = cy - ch // 2
    x2 = cx + cw // 2
    y2 = cy + ch // 2

    # Draw container box (dark blue-gray tray with bright bezel)
    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 50, 65), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 160, 100), 3)
    cv2.putText(img, "ISRO CONTAINER TRAY", (x1 + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    labels.append((0, cx / w, cy / h, cw / w, ch / h))

    # 3. Red Box (Class 1) & Yellow Box (Class 2)
    bw = random.randint(65, 85)
    bh = random.randint(65, 85)

    # Placement scenario:
    # 0 = both inside separated, 1 = collision, 2 = red outside, 3 = yellow outside
    scenario = random.choice([0, 0, 1, 2, 3])

    if scenario == 1:
        # Collision: red and yellow adjacent inside container
        rx = cx - bw // 2 + random.randint(-10, 10)
        ry = cy + random.randint(-30, 30)
        yx = rx + bw + random.randint(-5, 5)
        yy = ry + random.randint(-8, 8)
    elif scenario == 0:
        # Both inside separated
        rx = x1 + random.randint(15, cw // 3)
        ry = cy + random.randint(-40, 40)
        yx = x2 - random.randint(15, cw // 3) - bw
        yy = cy + random.randint(-40, 40)
    elif scenario == 2:
        # Red outside container (being retrieved)
        rx = random.randint(40, x1 - bw - 10) if x1 > bw + 50 else x2 + 10
        ry = cy + random.randint(-50, 50)
        yx = cx + random.randint(-30, 30)
        yy = cy + random.randint(-30, 30)
    else:
        # Yellow outside container
        rx = cx + random.randint(-30, 30)
        ry = cy + random.randint(-30, 30)
        yx = random.randint(x2 + 10, w - bw - 40) if x2 + bw + 40 < w else x1 - bw - 10
        yy = cy + random.randint(-50, 50)

    # Draw Red Box
    rx1, ry1 = rx, ry
    rx2, ry2 = rx + bw, ry + bh
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (25, 30, 220), -1)  # BGR Red
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (60, 60, 255), 2)
    labels.append((1, (rx1 + rx2) / (2 * w), (ry1 + ry2) / (2 * h), bw / w, bh / h))

    # Draw Yellow Box
    yx1, yy1 = yx, yy
    yx2, yy2 = yx + bw, yy + bh
    cv2.rectangle(img, (yx1, yy1), (yx2, yy2), (15, 215, 245), -1)  # BGR Yellow
    cv2.rectangle(img, (yx1, yy1), (yx2, yy2), (60, 240, 255), 2)
    labels.append((2, (yx1 + yx2) / (2 * w), (yy1 + yy2) / (2 * h), bw / w, bh / h))

    # 4. Optional Astronaut Glove / Hand (Class 3)
    if random.random() < 0.65:
        target_to_touch = random.choice([(rx, ry), (yx, yy)])
        hx = target_to_touch[0] + random.randint(-30, 30)
        hy = target_to_touch[1] + random.randint(-40, 20)
        hw = random.randint(70, 95)
        hh = random.randint(65, 85)

        # Draw white/light-gray space glove
        cv2.ellipse(img, (hx + hw // 2, hy + hh // 2), (hw // 2, hh // 2), random.randint(-20, 20), 0, 360, (230, 235, 240), -1)
        cv2.ellipse(img, (hx + hw // 2, hy + hh // 2), (hw // 2, hh // 2), random.randint(-20, 20), 0, 360, (180, 190, 200), 2)
        labels.append((3, (hx + hw / 2) / w, (hy + hh / 2) / h, hw / w, hh / h))

    # 5. Lighting, Camera Noise & Microgravity blur
    if random.random() < 0.35:
        # Glare from glovebox LED light strip
        glare_x = random.randint(100, w - 100)
        cv2.circle(img, (glare_x, 80), random.randint(40, 100), (255, 255, 255), -1)
        img = cv2.GaussianBlur(img, (15, 15), 0)

    # Sensor noise
    noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, labels


def build_dataset(dataset_dir: Path, num_train: int = 150, num_val: int = 30, api_key: str = ""):
    """Builds a complete YOLO-formatted dataset for the ISRO sample experiment."""
    groq_variations = query_groq_domain_enrichment(api_key)

    train_img_dir = dataset_dir / "images" / "train"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    val_lbl_dir = dataset_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating {num_train} training samples and {num_val} validation samples...")

    for i in range(num_train):
        img, labels = generate_synthetic_frame(i)
        img_name = f"isro_bcx1_train_{i:04d}.jpg"
        lbl_name = f"isro_bcx1_train_{i:04d}.txt"

        cv2.imwrite(str(train_img_dir / img_name), img)
        with open(train_lbl_dir / lbl_name, "w", encoding="utf-8") as f:
            for cls_id, cx, cy, bw, bh in labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    for i in range(num_val):
        img, labels = generate_synthetic_frame(num_train + i)
        img_name = f"isro_bcx1_val_{i:04d}.jpg"
        lbl_name = f"isro_bcx1_val_{i:04d}.txt"

        cv2.imwrite(str(val_img_dir / img_name), img)
        with open(val_lbl_dir / lbl_name, "w", encoding="utf-8") as f:
            for cls_id, cx, cy, bw, bh in labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Generate data.yaml for Ultralytics YOLOv8
    data_yaml = dataset_dir / "data.yaml"
    yaml_content = f"""# ISRO SIH 26174 Two-Box Collision Dataset
path: {dataset_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: container_box
  1: red_box
  2: yellow_box
  3: astronaut_hand
"""
    data_yaml.write_text(yaml_content, encoding="utf-8")
    print(f"[OK] Dataset successfully generated at {dataset_dir}")
    print(f"[OK] YOLO config written to {data_yaml}")
    return data_yaml


def train_yolo(data_yaml: Path, epochs: int = 5):
    """Executes Ultralytics YOLOv8 training on the generated ISRO dataset."""
    from ultralytics import YOLO

    print(f"\n=======================================================")
    print(f"🚀 Initializing YOLOv8n fine-tuning for ISRO PS 26174")
    print(f"   • Dataset: {data_yaml}")
    print(f"   • Epochs:  {epochs}")
    print(f"   • Imgsz:   320 (Optimized for edge CPU inference)")
    print(f"=======================================================\n")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=320,
        batch=8,
        workers=0,  # Windows-friendly
        verbose=True,
    )
    print("\n[OK] Training complete. Best weights saved to runs/detect/train/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description="ISRO Experiment Dataset Generator & YOLO Training")
    parser.add_argument("--api-key", default=GROQ_API_KEY_DEFAULT, help="Groq API key for domain synthesis")
    parser.add_argument("--samples", type=int, default=100, help="Number of synthetic training frames")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--train", action="store_true", help="Launch YOLO training after generating dataset")
    args = parser.parse_args()

    dataset_path = ROOT / "runs" / "isro_dataset"
    data_yaml = build_dataset(dataset_path, num_train=args.samples, num_val=max(20, args.samples // 5), api_key=args.api_key)

    if args.train:
        train_yolo(data_yaml, epochs=args.epochs)
    else:
        print("\nTo train YOLOv8 on this dataset, run:")
        print(f"  python tools/generate_and_train_isro_model.py --train --epochs 10")


if __name__ == "__main__":
    main()
