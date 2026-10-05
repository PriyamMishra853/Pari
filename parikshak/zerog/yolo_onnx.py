"""YOLOv8 detect / pose on ONNX Runtime, without PyTorch or ultralytics.

Same pre/post-processing as ultralytics for a static 320x320 export (letterbox
with grey 114 padding, conf 0.25, class-aware NMS at IoU 0.7), and results
exposed through the tiny subset of the ultralytics Results API the trackers
use: res.boxes[i].xyxy[0] / .conf[0] / .cls[0], res.keypoints.data[i].
That keeps the server image free of the ~1.5 GB torch stack.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

try:
    import importlib

    cv2 = importlib.import_module("cv2")
except ImportError:  # vision stack not installed: the engine still imports
    cv2 = None  # type: ignore

_MAX_WH = 7680.0  # class offset for class-aware NMS (ultralytics convention)


class _Arr:
    """numpy array with the .cpu().numpy() shape ultralytics callers expect."""

    def __init__(self, a: np.ndarray) -> None:
        self.a = a

    def cpu(self) -> "_Arr":
        return self

    def numpy(self) -> np.ndarray:
        return self.a

    def __getitem__(self, i):
        return _Arr(self.a[i])

    def __len__(self) -> int:
        return len(self.a)

    @property
    def shape(self):
        return self.a.shape


class _Box:
    __slots__ = ("xyxy", "conf", "cls")

    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = np.asarray([xyxy], dtype=np.float32)
        self.conf = np.asarray([conf], dtype=np.float32)
        self.cls = np.asarray([cls], dtype=np.float32)


class _Keypoints:
    def __init__(self, data: np.ndarray) -> None:
        self.data = _Arr(data)

    def __len__(self) -> int:
        return len(self.data)


class _Result:
    def __init__(self, boxes: list[_Box], keypoints: np.ndarray | None = None) -> None:
        self.boxes = boxes
        self.keypoints = _Keypoints(keypoints if keypoints is not None else np.zeros((0, 17, 3), np.float32))


class OnnxYolo:
    def __init__(self, path: str | Path, task: str = "detect", conf: float = 0.25, iou: float = 0.7) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.imgsz = int(shape[2]) if isinstance(shape[2], int) else 320
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta["names"]) if "names" in meta else {i: str(i) for i in range(80)}
        self.task = meta.get("task", task)
        self.conf, self.iou = conf, iou
        self.ckpt_path = str(path)

    # ultralytics-style call: model(frame, imgsz=..., verbose=...)[0]
    def __call__(self, frame: np.ndarray, imgsz: int | None = None, verbose: bool = False, **_):
        return [self.predict(frame)]

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        s = self.imgsz
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (s - nw) / 2.0, (s - nh) / 2.0
        if (w, h) != (nw, nh):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, left, top

    def predict(self, frame: np.ndarray) -> _Result:
        img, r, padx, pady = self._letterbox(frame)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0].T  # (N, 4+nc) or (N, 5+51)
        boxes_c = out[:, :4]
        if self.task == "pose":
            scores = out[:, 4]
            cls = np.zeros(len(out), dtype=np.int64)
            kpts = out[:, 5:]
        else:
            cls_scores = out[:, 4:]
            cls = cls_scores.argmax(1)
            scores = cls_scores[np.arange(len(out)), cls]
            kpts = None
        keep = scores >= self.conf
        if not np.any(keep):
            return _Result([], np.zeros((0, 17, 3), np.float32) if self.task == "pose" else None)
        boxes_c, scores, cls = boxes_c[keep], scores[keep], cls[keep]
        if kpts is not None:
            kpts = kpts[keep]
        xyxy = np.empty_like(boxes_c)
        xyxy[:, 0] = boxes_c[:, 0] - boxes_c[:, 2] / 2
        xyxy[:, 1] = boxes_c[:, 1] - boxes_c[:, 3] / 2
        xyxy[:, 2] = boxes_c[:, 0] + boxes_c[:, 2] / 2
        xyxy[:, 3] = boxes_c[:, 1] + boxes_c[:, 3] / 2
        off = cls[:, None].astype(np.float32) * _MAX_WH
        nb = xyxy + off
        idx = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in nb],
                               scores.astype(float).tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)[:300] if len(idx) else np.array([], dtype=int)
        idx = idx[np.argsort(-scores[idx])] if len(idx) else idx
        h, w = frame.shape[:2]
        res_boxes: list[_Box] = []
        for i in idx:
            b = xyxy[i].copy()
            b[[0, 2]] = ((b[[0, 2]] - padx) / r).clip(0, w)
            b[[1, 3]] = ((b[[1, 3]] - pady) / r).clip(0, h)
            res_boxes.append(_Box(b, float(scores[i]), int(cls[i])))
        kp_out = None
        if kpts is not None:
            kp = kpts[idx].reshape(-1, 17, 3).copy()
            kp[:, :, 0] = (kp[:, :, 0] - padx) / r
            kp[:, :, 1] = (kp[:, :, 1] - pady) / r
            kp[:, :, 0] = kp[:, :, 0].clip(0, w)  # clipped to the image, as ultralytics does
            kp[:, :, 1] = kp[:, :, 1].clip(0, h)
            kp_out = kp.astype(np.float32)
        return _Result(res_boxes, kp_out)
