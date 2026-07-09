"""
Local HTTP bridge for Pallet Sight web UI -> Intel RealSense D457 depth pipeline.

Run this on the Windows PC that has the D457 connected:

    python realsense_web_bridge.py --segmentation-backend depth

The browser cannot access `pyrealsense2` depth frames directly. This bridge owns
the RealSense pipeline, aligns depth to color, applies the ROI from the web UI,
segments candidate boxes, and returns measured camera-space Z in millimeters.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import cv2
import numpy as np

from depalletizing_realsense_d457 import (
    Detection,
    apply_roi,
    build_detections,
    choose_highest_detection,
    clean_mask,
    load_segmentation_model,
    non_max_mask_filter,
    segment_boxes,
    segment_boxes_from_depth,
    setup_realsense,
    stop_real_resources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pallet Sight RealSense web bridge")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--width", type=int, default=640, help="RealSense stream width")
    parser.add_argument("--height", type=int, default=480, help="RealSense stream height")
    parser.add_argument("--fps", type=int, default=30, help="RealSense stream FPS")
    parser.add_argument(
        "--segmentation-backend",
        choices=["depth", "fastsam", "yolo", "auto"],
        default="depth",
        help="Use depth-edge for no-AI depth segmentation, or FastSAM/YOLO masks with RealSense depth.",
    )
    parser.add_argument("--model", default="FastSAM-s.pt", help="FastSAM or YOLO segmentation weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Segmentation inference size")
    parser.add_argument("--conf", type=float, default=0.35, help="Segmentation confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="Segmentation NMS IoU threshold")
    parser.add_argument("--min-area", type=int, default=2000, help="Minimum mask area in pixels")
    parser.add_argument("--tie-depth-mm", type=float, default=5.0, help="Height tie threshold in millimeters")
    parser.add_argument("--dimension-scale", type=float, default=1.13, help="Calibration factor applied to measured length/width")
    parser.add_argument("--device", default=None, help="Ultralytics device, for example 0 or cpu")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when supported")
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def percent_roi_to_pixels(payload_roi: dict[str, Any], frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x_pct = clamp(float(payload_roi.get("x", 0.0)), 0.0, 100.0)
    y_pct = clamp(float(payload_roi.get("y", 0.0)), 0.0, 100.0)
    w_pct = clamp(float(payload_roi.get("w", 100.0)), 1.0, 100.0 - x_pct)
    h_pct = clamp(float(payload_roi.get("h", 100.0)), 1.0, 100.0 - y_pct)
    x = int(round((x_pct / 100.0) * frame_w))
    y = int(round((y_pct / 100.0) * frame_h))
    w = max(12, int(round((w_pct / 100.0) * frame_w)))
    h = max(12, int(round((h_pct / 100.0) * frame_h)))
    return x, y, min(w, frame_w - x), min(h, frame_h - y)


def rect_to_percent(rect: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> dict[str, float]:
    x, y, w, h = rect
    frame_h, frame_w = frame_shape[:2]
    return {
        "x": (x / frame_w) * 100.0,
        "y": (y / frame_h) * 100.0,
        "w": (w / frame_w) * 100.0,
        "h": (h / frame_h) * 100.0,
    }


def detection_rect(detection: Detection) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(detection.contour)
    return int(x), int(y), int(w), int(h)


def deproject_with_intrinsics(intrinsics: Any, pixel: tuple[float, float], depth_m: float) -> np.ndarray:
    x = (pixel[0] - intrinsics.ppx) / intrinsics.fx * depth_m
    y = (pixel[1] - intrinsics.ppy) / intrinsics.fy * depth_m
    return np.array([x, y, depth_m], dtype=np.float64) * 1000.0


def dimensions_from_detection(detection: Detection, intrinsics: Any, dimension_scale: float) -> dict[str, int]:
    depth_m = float(detection.avg_depth_m)
    rect = cv2.minAreaRect(detection.contour)
    points = cv2.boxPoints(rect)
    world_points = [deproject_with_intrinsics(intrinsics, (float(px), float(py)), depth_m) for px, py in points]
    side_lengths = [
        float(np.linalg.norm(world_points[(idx + 1) % 4][:2] - world_points[idx][:2]))
        for idx in range(4)
    ]
    dim_x = float(np.mean([side_lengths[0], side_lengths[2]])) * float(dimension_scale)
    dim_y = float(np.mean([side_lengths[1], side_lengths[3]])) * float(dimension_scale)
    return {
        "widthMm": int(round(min(dim_x, dim_y))),
        "lengthMm": int(round(max(dim_x, dim_y))),
        "dimXmm": int(round(dim_x)),
        "dimYmm": int(round(dim_y)),
    }


def reject_bright_support_surface(color_bgr: np.ndarray, detection: Detection) -> bool:
    mask = detection.mask > 0
    if int(np.count_nonzero(mask)) < 1:
        return True
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1][mask]
    val = hsv[:, :, 2][mask]
    mean_sat = float(np.mean(sat))
    mean_val = float(np.mean(val))
    x, y, w, h = detection_rect(detection)
    frame_h, frame_w = color_bgr.shape[:2]
    box_area_ratio = (w * h) / float(frame_w * frame_h)
    return mean_sat < 28.0 and mean_val > 118.0 and box_area_ratio > 0.08


def segment_color_candidates(color_bgr: np.ndarray, roi_mask: np.ndarray, min_area_px: int) -> list[np.ndarray]:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    dark_object = (v < 118).astype(np.uint8) * 255
    cardboard_or_colored = ((s > 42) & (v > 55) & (v < 245)).astype(np.uint8) * 255
    candidate = cv2.bitwise_or(dark_object, cardboard_or_colored)
    candidate = cv2.bitwise_and(candidate, roi_mask)
    candidate = clean_mask(candidate)
    candidate = cv2.dilate(candidate, np.ones((11, 11), np.uint8), iterations=1)
    candidate = cv2.bitwise_and(candidate, roi_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    masks: list[np.ndarray] = []
    roi_area = max(1, int(np.count_nonzero(roi_mask)))
    roi_points = cv2.findNonZero(roi_mask)
    if roi_points is None:
        return []
    roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(roi_points)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(80, int(min_area_px * 0.35)):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area > roi_area * 0.55:
            continue
        if x <= roi_x + 2 or y <= roi_y + 2 or x + w >= roi_x + roi_w - 3 or y + h_box >= roi_y + roi_h - 3:
            continue
        mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
        mask[labels == label] = 255
        masks.append(clean_mask(mask))
    return non_max_mask_filter(masks, max_overlap=0.88)


def has_depth_contrast(mask: np.ndarray, depth_raw: np.ndarray, roi_mask: np.ndarray, depth_scale: float) -> bool:
    kernel = np.ones((17, 17), np.uint8)
    ring = cv2.dilate(mask, kernel, iterations=1)
    ring = cv2.bitwise_and(ring, roi_mask)
    ring[mask > 0] = 0
    inside = depth_raw[mask > 0]
    outside = depth_raw[ring > 0]
    inside = inside[inside > 0]
    outside = outside[outside > 0]
    if inside.size < 30 or outside.size < 30:
        return False
    inside_mm = np.median(inside.astype(np.float32) * float(depth_scale) * 1000.0)
    outside_mm = np.median(outside.astype(np.float32) * float(depth_scale) * 1000.0)
    return bool(inside_mm < outside_mm - 3.0)


def detection_to_payload(
    detection: Detection,
    index: int,
    best_idx: Optional[int],
    frame_shape: tuple[int, int],
    intrinsics: Any,
    dimension_scale: float,
) -> dict[str, Any]:
    camera_xyz = detection.camera_xyz_mm if detection.camera_xyz_mm is not None else np.zeros(3)
    robot_xyz = detection.robot_xyz_mm if detection.robot_xyz_mm is not None else np.zeros(3)
    return {
        "id": index + 1,
        "isBest": index == best_idx,
        **rect_to_percent(detection_rect(detection), frame_shape),
        **dimensions_from_detection(detection, intrinsics, dimension_scale),
        "centerPx": [int(detection.center_px[0]), int(detection.center_px[1])],
        "z": int(round(detection.avg_depth_m * 1000.0)),
        "camera_xyz": [round(float(v), 1) for v in camera_xyz.tolist()],
        "robot_xyz": [round(float(v), 1) for v in robot_xyz.tolist()],
        "area": round(float(detection.area_px), 1),
        "color": "#def4a5",
        "source": "realsense-depth",
    }


def encode_frame_jpeg(color_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("Failed to encode RealSense color frame.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


class RealSenseBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.pipeline = None
        self.align = None
        self.depth_scale: Optional[float] = None
        self.intrinsics = None
        self.model = None
        self.model_type = "camera"

    def close(self) -> None:
        state = {"pipeline": self.pipeline, "align": self.align, "depth_scale": self.depth_scale, "intrinsics": self.intrinsics}
        stop_real_resources(state)
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.intrinsics = None

    def ensure_started(self) -> None:
        if self.pipeline is None or self.align is None:
            self.pipeline, self.align = setup_realsense(self.args.width, self.args.height, self.args.fps)
            depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

        if self.args.segmentation_backend == "depth":
            self.model = None
            self.model_type = "depth-edge"
        elif self.model is None:
            self.model, self.model_type = load_segmentation_model(self.args.model, self.args.segmentation_backend)

    def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.ensure_started()
            try:
                for _ in range(5):
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                    aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    raise RuntimeError("RealSense returned incomplete aligned frames.")

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                self.intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                roi = percent_roi_to_pixels(payload.get("roi", {}), color.shape[:2])
                color_roi, depth_roi, roi_mask = apply_roi(color, depth, roi)

                if self.model_type == "depth-edge":
                    masks = segment_boxes_from_depth(
                        depth_raw=depth_roi,
                        roi_mask=roi_mask,
                        depth_scale=float(self.depth_scale),
                        min_area_px=self.args.min_area,
                    )
                    if not masks:
                        masks = [
                            mask
                            for mask in segment_color_candidates(color_roi, roi_mask, self.args.min_area)
                            if has_depth_contrast(mask, depth_roi, roi_mask, float(self.depth_scale))
                        ]
                else:
                    masks = segment_boxes(
                        model=self.model,
                        model_type=str(self.model_type),
                        color_bgr=color_roi,
                        roi_mask=roi_mask,
                        imgsz=self.args.imgsz,
                        conf=self.args.conf,
                        iou=self.args.iou,
                        min_area_px=self.args.min_area,
                        device=self.args.device,
                        half=self.args.half,
                    )

                detections, best_idx = build_detections(
                    masks=masks,
                    depth_raw=depth_roi,
                    depth_scale=float(self.depth_scale),
                    intrinsics=self.intrinsics,
                    tie_depth_mm=self.args.tie_depth_mm,
                    frame_center=(color.shape[1] / 2.0, color.shape[0] / 2.0),
                )
                detections = [
                    detection
                    for detection in detections
                    if not reject_bright_support_surface(color_roi, detection)
                ]
                best_idx = choose_highest_detection(
                    detections,
                    self.args.tie_depth_mm,
                    (color.shape[1] / 2.0, color.shape[0] / 2.0),
                )

                result_detections = [
                    detection_to_payload(detection, index, best_idx, color.shape[:2], self.intrinsics, self.args.dimension_scale)
                    for index, detection in enumerate(detections)
                ]
                return {
                    "ok": True,
                    "message": f"RealSense depth detected {len(result_detections)} box(es)",
                    "modelType": self.model_type,
                    "bestId": None if best_idx is None else best_idx + 1,
                    "roiPixels": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
                    "frame": encode_frame_jpeg(color_roi),
                    "detections": result_detections,
                }
            except Exception:
                self.close()
                raise

    def preview(self) -> dict[str, Any]:
        with self.lock:
            self.ensure_started()
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise RuntimeError("RealSense returned incomplete preview frames.")
                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                valid_depth = int(np.count_nonzero(depth))
                return {
                    "ok": True,
                    "message": "RealSense RGB/depth preview live",
                    "frame": encode_frame_jpeg(color),
                    "width": int(color.shape[1]),
                    "height": int(color.shape[0]),
                    "validDepthPixels": valid_depth,
                }
            except Exception:
                self.close()
                raise


def make_handler(bridge: RealSenseBridge):
    class BridgeHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self._send_json(200, {"ok": True})

        def do_GET(self) -> None:
            if self.path == "/api/health":
                self._send_json(200, {"ok": True, "message": "RealSense bridge is running"})
                return
            if self.path == "/api/preview":
                try:
                    self._send_json(200, bridge.preview())
                except Exception as exc:
                    self._send_json(503, {"ok": False, "message": str(exc)})
                return
            self._send_json(404, {"ok": False, "message": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/api/capture":
                self._send_json(404, {"ok": False, "message": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length) if content_length else b"{}"
                payload = json.loads(raw_body.decode("utf-8") or "{}")
                self._send_json(200, bridge.capture(payload))
            except Exception as exc:
                self._send_json(503, {"ok": False, "message": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return BridgeHandler


def main() -> int:
    args = parse_args()
    bridge = RealSenseBridge(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bridge))
    print(f"Pallet Sight RealSense bridge listening on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
