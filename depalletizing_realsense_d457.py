"""
RealSense D457 mixed-case depalletizing vision pipeline.

This script is designed for an Intel RealSense D457 mounted eye-to-hand,
top-down over a configurable pallet footprint. It aligns depth to color, lets the
operator select the pallet ROI, segments tightly packed boxes using a
zero-shot segmentation model, selects the highest box, deprojects its pick
center to 3D camera coordinates, and transforms that point into robot base
coordinates through a configurable homogeneous transform.

Install:
    pip install -r requirements.txt

Recommended model:
    FastSAM-s.pt or FastSAM-x.pt from Ultralytics. The script defaults to
    FastSAM-s.pt for speed. YOLOv8 segmentation weights can also be supplied.
    Use --segmentation-backend depth for a non-AI depth-edge fallback.

Controls:
    r  - reselect pallet ROI
    s  - save current ROI to roi_config.json
    l  - load ROI from roi_config.json
    q  - quit

Notes:
    - D457 product specs list depth FOV as H:87 deg, V:58 deg (+/-3 deg).
      Some product overview material lists RGB FOV as approximately
      H:90 deg, V:65 deg (+/-3 deg). Because depth is required for picking,
      the height recommendation is computed from the conservative depth FOV.
    - RealSense depth values and rs2_deproject_pixel_to_point expect meters.
      Robot output is shown in millimeters.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "Missing OpenCV. Install it with: pip install opencv-python"
    ) from exc

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from ultralytics import FastSAM, YOLO
except ImportError:
    FastSAM = None
    YOLO = None


DEFAULT_PALLET_WIDTH_MM = 1100.0
DEFAULT_PALLET_DEPTH_MM = 1100.0
D457_DEPTH_FOV_H_DEG = 87.0
D457_DEPTH_FOV_V_DEG = 58.0
D457_FOV_TOLERANCE_DEG = 3.0
D457_MIN_OPERATING_RANGE_MM = 600.0
DEFAULT_ROI_FILE = Path("roi_config.json")
DEMO_TOGGLE_RECT = (24, 118, 170, 32)
CAPTURE_RECT = (24, 618, 106, 32)
CONFIRM_RECT = (140, 618, 106, 32)
NEXT_RECT = (256, 618, 96, 32)
REDRAG_RECT = (24, 660, 154, 32)
EXIT_RECT = (190, 660, 78, 32)
DEFAULT_DEMO_ROI = (390, 70, 820, 610)
PANEL_WIDTH = 380


@dataclass(frozen=True)
class CameraMountRecommendation:
    min_height_mm: float
    recommended_height_mm: float
    conservative_hfov_deg: float
    conservative_vfov_deg: float
    note: str


@dataclass
class Detection:
    mask: np.ndarray
    contour: np.ndarray
    center_px: tuple[int, int]
    avg_depth_m: float
    area_px: float
    camera_xyz_mm: Optional[np.ndarray] = None
    robot_xyz_mm: Optional[np.ndarray] = None


class GracefulShutdown:
    def __init__(self) -> None:
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._request_stop)
        signal.signal(signal.SIGTERM, self._request_stop)

    def _request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True


def calculate_mounting_height(
    pallet_width_mm: float = DEFAULT_PALLET_WIDTH_MM,
    pallet_depth_mm: float = DEFAULT_PALLET_DEPTH_MM,
    hfov_deg: float = D457_DEPTH_FOV_H_DEG,
    vfov_deg: float = D457_DEPTH_FOV_V_DEG,
    tolerance_deg: float = D457_FOV_TOLERANCE_DEG,
    margin: float = 0.15,
) -> CameraMountRecommendation:
    """Calculate minimum and recommended camera height above the pallet top."""

    conservative_hfov = hfov_deg - tolerance_deg
    conservative_vfov = vfov_deg - tolerance_deg
    half_width = pallet_width_mm / 2.0
    half_depth = pallet_depth_mm / 2.0
    required_by_width = half_width / math.tan(math.radians(conservative_hfov / 2.0))
    required_by_depth = half_depth / math.tan(math.radians(conservative_vfov / 2.0))
    min_height = max(required_by_width, required_by_depth, D457_MIN_OPERATING_RANGE_MM)
    recommended = min_height * (1.0 + margin)
    note = (
        "Use the conservative depth FOV because depth must cover every pick "
        "candidate. Add margin for installation tolerances, pallet skew, and "
        "ROI cropping."
    )
    return CameraMountRecommendation(
        min_height_mm=min_height,
        recommended_height_mm=recommended,
        conservative_hfov_deg=conservative_hfov,
        conservative_vfov_deg=conservative_vfov,
        note=note,
    )


def print_mounting_recommendation(
    recommendation: CameraMountRecommendation,
    pallet_width_mm: float,
    pallet_depth_mm: float,
) -> None:
    print("\n=== Intel RealSense D457 Mounting Height Recommendation ===")
    print(
        "D457 depth FOV used: "
        f"H={D457_DEPTH_FOV_H_DEG:.1f} deg, V={D457_DEPTH_FOV_V_DEG:.1f} deg "
        f"(conservative: H={recommendation.conservative_hfov_deg:.1f}, "
        f"V={recommendation.conservative_vfov_deg:.1f})"
    )
    print(f"Pallet footprint: {pallet_width_mm:.0f} mm x {pallet_depth_mm:.0f} mm")
    print(f"Minimum Z from pallet top: {recommendation.min_height_mm:.0f} mm")
    print(f"Recommended Z with 15% margin: {recommendation.recommended_height_mm:.0f} mm")
    print(f"Note: {recommendation.note}\n")


def transform_camera_to_robot(camera_coords_mm: np.ndarray) -> np.ndarray:
    """
    Convert camera-space XYZ coordinates to robot-base XYZ coordinates.

    Eye-to-hand calibration guide
    =============================

    Goal
    ----
    Fill `T_ROBOT_FROM_CAMERA` with the rigid transform that maps a point
    measured by the D457 camera into the robot base frame:

        P_robot = T_robot_from_camera @ [Xc, Yc, Zc, 1]^T

    Coordinate frames
    -----------------
    RealSense camera coordinates follow the pinhole camera convention:
    - +Xc points right in the image.
    - +Yc points down in the image.
    - +Zc points forward from the camera lens toward the scene.

    For a top-down installation, +Zc typically points downward toward the
    pallet, while many robot bases use +Zr upward. That means the transform
    often contains an axis flip between camera Z and robot Z. Do not guess the
    signs from intuition only; solve them from measured calibration pairs.

    Recommended eye-to-hand workflow
    --------------------------------
    1. Rigidly mount the D457. Lock focus, exposure strategy, resolution, and
       robot/camera fixtures before collecting data.
    2. Place a calibration target or a small, precisely touchable fiducial on
       the pallet plane. Charuco, AprilTag boards, or a machined puck with a
       visually detectable center work well.
    3. Move the robot TCP to at least 9-15 known target points distributed
       across the pallet volume: corners, edges, center, and several heights.
       Avoid using only coplanar points if the robot must pick boxes of many
       heights; 3D spread improves the transform.
    4. For every point, record:
       - Camera measurement: [Xc, Yc, Zc] from `rs2_deproject_pixel_to_point`.
       - Robot measurement: [Xr, Yr, Zr] from the robot base frame, ideally the
         TCP position when touching the same physical point.
    5. Solve the rigid transform using a least-squares Kabsch/Umeyama fit or
       OpenCV hand-eye/pose tools. Validate residual error in millimeters.
    6. Inspect the resulting rotation matrix. In a top-down cell it is normal
       for camera +Z to map approximately to robot -Z if robot +Z is up. Camera
       +X/+Y may also be swapped or negated depending on camera yaw.
    7. Enter the final 4x4 matrix below. Keep units in millimeters because this
       application reports camera coordinates in millimeters.
    8. Validate with independent checkpoints before enabling automatic picks.
       A practical acceptance test is: selected visual point, robot approach
       point, and touched physical point agree within the required grasp
       tolerance across the full pallet.

    Production cautions
    -------------------
    - Recalibrate after moving the camera, robot base, pallet fixture, lens, or
      changing any mechanical bracket.
    - Keep calibration data and the final matrix version-controlled with date,
      operator, robot, camera serial number, and residual error report.
    - Add the gripper/tool-center offset after this base transform, not inside
      the camera transform, unless your calibration target was touched with the
      same TCP definition used for production picks.
    """

    T_ROBOT_FROM_CAMERA = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_h = np.array(
        [camera_coords_mm[0], camera_coords_mm[1], camera_coords_mm[2], 1.0],
        dtype=np.float64,
    )
    robot_h = T_ROBOT_FROM_CAMERA @ camera_h
    return robot_h[:3]


def require_realsense() -> None:
    if rs is None:
        raise RuntimeError("Missing pyrealsense2. Install it with: pip install pyrealsense2")


def require_ultralytics() -> None:
    if FastSAM is None or YOLO is None:
        raise RuntimeError("Missing ultralytics. Install it with: pip install ultralytics")


def setup_realsense(width: int, height: int, fps: int):
    require_realsense()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not start RealSense pipeline. Check D457 connection, power, "
            "GMSL/USB adapter, firmware, and whether another process owns it."
        ) from exc

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    if depth_sensor.supports(rs.option.laser_power):
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        depth_sensor.set_option(rs.option.laser_power, laser_range.max)

    align = rs.align(rs.stream.color)
    return pipeline, align


def warmup_and_get_frame(
    pipeline: rs.pipeline,
    align: rs.align,
    warmup_frames: int = 20,
) -> tuple[np.ndarray, np.ndarray, rs.video_stream_profile, float]:
    for _ in range(warmup_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

    aligned_depth = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not aligned_depth or not color_frame:
        raise RuntimeError("RealSense produced incomplete frames during warmup.")

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(aligned_depth.get_data())
    depth_scale = (
        pipeline.get_active_profile()
        .get_device()
        .first_depth_sensor()
        .get_depth_scale()
    )
    color_profile = color_frame.profile.as_video_stream_profile()
    return color_image, depth_image, color_profile, depth_scale


def select_roi(color_image: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    display = color_image.copy()
    cv2.putText(
        display,
        "Draw pallet ROI, press ENTER/SPACE to confirm, ESC to cancel",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    roi = cv2.selectROI("ROI Setup - Select Pallet Area", display, False, False)
    cv2.destroyWindow("ROI Setup - Select Pallet Area")
    x, y, w, h = [int(v) for v in roi]
    if w <= 10 or h <= 10:
        return None
    return x, y, w, h


def save_roi(path: Path, roi: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> None:
    payload = {
        "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
        "frame_width": frame_shape[1],
        "frame_height": frame_shape[0],
        "saved_at_unix": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_roi(path: Path, frame_shape: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    roi_data = payload["roi"]
    roi = (
        int(roi_data["x"]),
        int(roi_data["y"]),
        int(roi_data["w"]),
        int(roi_data["h"]),
    )
    if not validate_roi(roi, frame_shape):
        return None
    return roi


def validate_roi(roi: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> bool:
    x, y, w, h = roi
    frame_h, frame_w = frame_shape[:2]
    return x >= 0 and y >= 0 and w > 10 and h > 10 and x + w <= frame_w and y + h <= frame_h


def apply_roi(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    roi: Optional[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if roi is None:
        mask = np.ones(depth_image.shape[:2], dtype=np.uint8) * 255
        return color_image, depth_image, mask

    x, y, w, h = roi
    mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    color_masked = color_image.copy()
    depth_masked = depth_image.copy()
    color_masked[mask == 0] = (18, 18, 18)
    depth_masked[mask == 0] = 0
    return color_masked, depth_masked, mask


def load_segmentation_model(model_path: str, backend: str):
    if backend == "depth":
        return None, "depth-edge"
    require_ultralytics()
    model_name = Path(model_path).name.lower()
    if backend == "fastsam" or (backend == "auto" and "fastsam" in model_name):
        return FastSAM(model_path), "fastsam"
    if backend == "yolo" or backend == "auto":
        return YOLO(model_path), "yolo-seg"
    return YOLO(model_path), "yolo-seg"


def segment_boxes(
    model,
    model_type: str,
    color_bgr: np.ndarray,
    roi_mask: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    min_area_px: int,
    device: Optional[str],
    half: bool,
) -> list[np.ndarray]:
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    results = model(
        rgb,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        retina_masks=False,
        stream=False,
        verbose=False,
    )

    masks: list[np.ndarray] = []
    if not results:
        return masks

    result = results[0]
    if result.masks is None:
        return masks

    mask_data = result.masks.data.detach().cpu().numpy()
    frame_h, frame_w = color_bgr.shape[:2]
    for raw_mask in mask_data:
        mask = (raw_mask > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != (frame_h, frame_w):
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = clean_mask(mask)
        area = int(cv2.countNonZero(mask))
        if area >= min_area_px:
            masks.append(mask)

    return non_max_mask_filter(masks, max_overlap=0.88)


def segment_boxes_from_depth(
    depth_raw: np.ndarray,
    roi_mask: np.ndarray,
    depth_scale: float,
    min_area_px: int,
    edge_threshold_mm: float = 35.0,
) -> list[np.ndarray]:
    """Segment candidate box top regions from depth discontinuities.

    This is a non-teaching fallback. It is useful when the scene has clear
    height/depth breaks, but zero-shot RGB segmentation should remain the
    preferred mode for tightly packed boxes of similar height.
    """

    valid = (depth_raw > 0) & (roi_mask > 0)
    if int(np.count_nonzero(valid)) < min_area_px:
        return []

    depth_mm = depth_raw.astype(np.float32) * float(depth_scale) * 1000.0
    valid_depth = depth_mm[valid]
    low, high = np.percentile(valid_depth, [2, 98])
    background_depth = float(np.percentile(valid_depth, 92))
    foreground_margin = max(4.0, edge_threshold_mm * 0.15)
    clipped = np.clip(depth_mm, low, high)
    clipped[~valid] = high
    smooth = cv2.medianBlur(clipped.astype(np.float32), 5)
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    height_foreground = valid & (depth_mm < (background_depth - foreground_margin))
    if int(np.count_nonzero(height_foreground)) < min_area_px:
        height_foreground = valid
    interior = ((gradient < edge_threshold_mm) & height_foreground).astype(np.uint8) * 255
    interior = cv2.bitwise_and(interior, roi_mask)
    interior = clean_mask(interior)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(interior, 8)
    masks: list[np.ndarray] = []
    frame_h, frame_w = depth_raw.shape[:2]
    roi_points = cv2.findNonZero(roi_mask)
    if roi_points is None:
        return []
    roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(roi_points)
    roi_area = int(np.count_nonzero(roi_mask))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_frame_border = x <= 1 or y <= 1 or x + w >= frame_w - 1 or y + h >= frame_h - 1
        touches_roi_border = (
            x <= roi_x + 3
            or y <= roi_y + 3
            or x + w >= roi_x + roi_w - 4
            or y + h >= roi_y + roi_h - 4
        )
        too_large = area > int(roi_area * 0.45) or (w > roi_w * 0.85 and h > roi_h * 0.85)
        boundary_surface = touches_roi_border and area > max(min_area_px * 2, int(roi_area * 0.08))
        if touches_frame_border or too_large or boundary_surface:
            continue
        mask = np.zeros(depth_raw.shape[:2], dtype=np.uint8)
        mask[labels == label] = 255
        masks.append(clean_mask(mask))

    return non_max_mask_filter(masks, max_overlap=0.88)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned


def non_max_mask_filter(masks: list[np.ndarray], max_overlap: float) -> list[np.ndarray]:
    if not masks:
        return []
    masks_sorted = sorted(masks, key=cv2.countNonZero, reverse=True)
    kept: list[np.ndarray] = []
    for mask in masks_sorted:
        area = cv2.countNonZero(mask)
        if area == 0:
            continue
        duplicate = False
        for existing in kept:
            intersection = cv2.countNonZero(cv2.bitwise_and(mask, existing))
            if intersection / float(area) > max_overlap:
                duplicate = True
                break
        if not duplicate:
            kept.append(mask)
    return kept


def build_detections(
    masks: Iterable[np.ndarray],
    depth_raw: np.ndarray,
    depth_scale: float,
    intrinsics: rs.intrinsics,
    tie_depth_mm: float,
    frame_center: tuple[float, float],
) -> tuple[list[Detection], Optional[int]]:
    detections: list[Detection] = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        valid_depth_m = robust_depth_for_mask(depth_raw, mask, depth_scale)
        if valid_depth_m is None:
            continue

        require_realsense()
        camera_xyz_m = rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [float(center_x), float(center_y)],
            float(valid_depth_m),
        )
        camera_xyz_mm = np.array(camera_xyz_m, dtype=np.float64) * 1000.0
        robot_xyz_mm = transform_camera_to_robot(camera_xyz_mm)

        detections.append(
            Detection(
                mask=mask,
                contour=contour,
                center_px=(center_x, center_y),
                avg_depth_m=valid_depth_m,
                area_px=area_px,
                camera_xyz_mm=camera_xyz_mm,
                robot_xyz_mm=robot_xyz_mm,
            )
        )

    best_idx = choose_highest_detection(detections, tie_depth_mm, frame_center)
    return detections, best_idx


def robust_depth_for_mask(
    depth_raw: np.ndarray,
    mask: np.ndarray,
    depth_scale: float,
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
) -> Optional[float]:
    raw_values = depth_raw[mask > 0]
    raw_values = raw_values[raw_values > 0]
    if raw_values.size < 50:
        return None
    depth_m = raw_values.astype(np.float32) * float(depth_scale)
    low, high = np.percentile(depth_m, [lower_percentile, upper_percentile])
    trimmed = depth_m[(depth_m >= low) & (depth_m <= high)]
    if trimmed.size < 50:
        return None
    return float(np.mean(trimmed))


def choose_highest_detection(
    detections: list[Detection],
    tie_depth_mm: float,
    frame_center: tuple[float, float],
) -> Optional[int]:
    if not detections:
        return None

    min_depth_m = min(d.avg_depth_m for d in detections)
    tied: list[tuple[int, Detection]] = [
        (idx, det)
        for idx, det in enumerate(detections)
        if abs(det.avg_depth_m - min_depth_m) * 1000.0 <= tie_depth_mm
    ]
    if len(tied) == 1:
        return tied[0][0]

    cx, cy = frame_center
    tied.sort(
        key=lambda item: math.hypot(item[1].center_px[0] - cx, item[1].center_px[1] - cy)
    )
    return tied[0][0]


def draw_operator_ui(
    frame: np.ndarray,
    roi: Optional[tuple[int, int, int, int]],
    detections: list[Detection],
    best_idx: Optional[int],
    fps: float,
    recommendation: CameraMountRecommendation,
    model_type: str,
    demo_enabled: bool,
    connection_status: str,
    capture_state: str,
    drag_roi: Optional[tuple[int, int, int, int]] = None,
    mouse_pos: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    out = frame.copy()
    colors = palette()

    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(out, (x, y), (x + w, y + h), (120, 210, 255), 2)
        cv2.putText(out, "PALLET ROI", (x + 10, max(y - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 210, 255), 2)
    if drag_roi is not None:
        x, y, w, h = drag_roi
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(out, "DRAG ROI", (x + 10, max(y - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for idx, det in enumerate(detections):
        color = colors[idx % len(colors)]
        is_best = idx == best_idx
        thickness = 4 if is_best else 2
        cv2.drawContours(out, [det.contour], -1, color, thickness, cv2.LINE_AA)
        label = f"BOX {idx + 1}  Z={det.avg_depth_m * 1000.0:.0f}mm"
        x, y = det.center_px
        cv2.circle(out, (x, y), 5 if is_best else 3, color, -1, cv2.LINE_AA)
        cv2.putText(out, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    if best_idx is not None:
        det = detections[best_idx]
        x, y = det.center_px
        cv2.drawMarker(out, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 34, 3, cv2.LINE_AA)
        cv2.circle(out, (x, y), 14, (0, 255, 255), 2, cv2.LINE_AA)

    panel_w = 380
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, out.shape[0]), (16, 22, 30), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)

    put_text(out, "Pallet Sight", (24, 36), 0.82, (245, 248, 252), 2)
    put_text(out, "Mixed case top-down vision", (24, 64), 0.50, (164, 176, 190), 1)
    draw_chip(out, (24, 88), "LIVE", (43, 185, 126))
    draw_chip(out, (92, 88), model_type.upper(), (82, 153, 255))
    draw_chip(out, (204, 88), f"{fps:4.1f} FPS", (245, 174, 66))
    draw_demo_toggle(out, demo_enabled, mouse_pos=mouse_pos)

    y0 = 176
    status_rows = [
        ("Camera", connection_status),
        ("Sequence", capture_state),
        ("ROI", "selected" if roi else "full frame"),
        ("Detections", str(len(detections))),
        ("Min mount Z", f"{recommendation.min_height_mm:.0f} mm"),
        ("Recommend Z", f"{recommendation.recommended_height_mm:.0f} mm"),
    ]
    for key, value in status_rows:
        put_text(out, key, (24, y0), 0.48, (150, 162, 176), 1)
        put_text(out, value, (150, y0), 0.55, (235, 240, 246), 1)
        y0 += 30

    y0 += 18
    put_text(out, "Pick Candidate", (24, y0), 0.62, (245, 248, 252), 2)
    y0 += 34
    if best_idx is None:
        put_text(out, "No valid pick target", (24, y0), 0.58, (120, 132, 146), 1)
    else:
        det = detections[best_idx]
        camera = det.camera_xyz_mm if det.camera_xyz_mm is not None else np.zeros(3)
        robot = det.robot_xyz_mm if det.robot_xyz_mm is not None else np.zeros(3)
        pick_rows = [
            ("Pixel", f"({det.center_px[0]}, {det.center_px[1]})"),
            ("Camera", f"X {camera[0]:.1f}  Y {camera[1]:.1f}"),
            ("", f"Z {camera[2]:.1f} mm"),
            ("Robot", f"X {robot[0]:.1f}  Y {robot[1]:.1f}"),
            ("", f"Z {robot[2]:.1f} mm"),
        ]
        for key, value in pick_rows:
            put_text(out, key, (24, y0), 0.45, (150, 162, 176), 1)
            put_text(out, value, (150, y0), 0.50, (245, 248, 252), 1)
            y0 += 22

    guide_y = 532
    put_text(out, "Walkthrough", (24, guide_y), 0.54, (245, 248, 252), 1)
    guide_lines = guide_for_state(capture_state)
    for idx, line in enumerate(guide_lines):
        put_text(out, line, (24, guide_y + 26 + idx * 20), 0.42, (164, 176, 190), 1)

    draw_button(out, CAPTURE_RECT, "CAPTURE", (82, 153, 255), enabled=True, mouse_pos=mouse_pos)
    draw_button(out, CONFIRM_RECT, "CONFIRM", (43, 185, 126), enabled=best_idx is not None, mouse_pos=mouse_pos)
    draw_button(out, NEXT_RECT, "NEXT", (245, 174, 66), enabled=len(detections) > 1, mouse_pos=mouse_pos)
    draw_button(out, REDRAG_RECT, "REDRAG ROI", (120, 210, 255), enabled=True, mouse_pos=mouse_pos)
    draw_button(out, EXIT_RECT, "EXIT", (235, 101, 101), enabled=True, mouse_pos=mouse_pos)

    footer_y = out.shape[0] - 16
    put_text(out, "Keys: c capture  n next  p confirm  r redrag  q exit", (24, footer_y), 0.38, (164, 176, 190), 1)
    if mouse_pos is not None and any(
        click_inside(mouse_pos, rect)
        for rect in (DEMO_TOGGLE_RECT, CAPTURE_RECT, CONFIRM_RECT, NEXT_RECT, REDRAG_RECT, EXIT_RECT)
    ):
        draw_pointing_hand(out, mouse_pos)
    return out


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_chip(
    image: np.ndarray,
    origin: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    w = text_size[0] + 24
    h = 26
    cv2.rectangle(image, (x, y), (x + w, y + h), (35, 44, 58), -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    cv2.circle(image, (x + 11, y + h // 2), 4, color, -1)
    put_text(image, label, (x + 20, y + 18), 0.45, (235, 240, 246), 1)


def draw_demo_toggle(
    image: np.ndarray,
    demo_enabled: bool,
    mouse_pos: Optional[tuple[int, int]] = None,
) -> None:
    x, y, w, h = DEMO_TOGGLE_RECT
    hovered = mouse_pos is not None and click_inside(mouse_pos, DEMO_TOGGLE_RECT)
    bg = (31, 97, 65) if demo_enabled else (61, 67, 77)
    accent = (43, 185, 126) if demo_enabled else (235, 101, 101)
    border = (255, 255, 255) if hovered else accent
    label = "DEMO ON" if demo_enabled else "DEMO OFF"
    cv2.rectangle(image, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), border, 2)
    draw_tiny_pointer(image, (x + 10, y + 10), border)
    knob_x = x + w - 27 if demo_enabled else x + 8
    cv2.circle(image, (knob_x + 8, y + h // 2), 8, accent, -1, cv2.LINE_AA)
    put_text(image, label, (x + 52, y + 21), 0.48, (245, 248, 252), 1)


def draw_button(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    enabled: bool,
    mouse_pos: Optional[tuple[int, int]] = None,
) -> None:
    x, y, w, h = rect
    hovered = enabled and mouse_pos is not None and click_inside(mouse_pos, rect)
    bg = (35, 44, 58) if enabled else (44, 48, 54)
    fg = (245, 248, 252) if enabled else (135, 143, 154)
    border = (255, 255, 255) if hovered else (color if enabled else (82, 88, 96))
    cv2.rectangle(image, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), border, 2)
    if enabled:
        draw_tiny_pointer(image, (x + 9, y + 11), border)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    tx = x + max(22, (w - text_size[0]) // 2 + 8)
    put_text(image, label, (tx, y + 22), 0.48, fg, 1)


def guide_for_state(capture_state: str) -> list[str]:
    if capture_state == "review pick":
        return [
            "1 Review frozen capture.",
            "2 Click box or NEXT.",
            "3 CONFIRM to proceed.",
        ]
    if capture_state == "confirmed":
        return [
            "Pick confirmed.",
            "Robot command can use",
            "shown XYZ coordinates.",
        ]
    if capture_state == "camera blocked":
        return [
            "Check D457 connection.",
            "Power/cable/firmware.",
            "CAPTURE retries start.",
        ]
    return [
        "1 Drag pallet ROI.",
        "2 Click CAPTURE once.",
        "3 Review before confirm.",
    ]


def draw_tiny_pointer(
    image: np.ndarray,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    pts = np.array([[x, y], [x, y + 15], [x + 4, y + 11], [x + 7, y + 18], [x + 10, y + 17], [x + 7, y + 10], [x + 13, y + 10]], dtype=np.int32)
    cv2.polylines(image, [pts], True, color, 1, cv2.LINE_AA)


def draw_pointing_hand(image: np.ndarray, point: tuple[int, int]) -> None:
    x, y = point
    cv2.circle(image, (x + 6, y + 8), 5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (x + 6, y + 8), (x + 6, y - 10), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(image, (x + 6, y - 10), (x + 16, y - 10), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(image, (x + 10, y + 4), (x + 18, y + 4), (255, 255, 255), 2, cv2.LINE_AA)


def click_inside(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    px, py = point
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def normalize_rect(
    start: tuple[int, int],
    end: tuple[int, int],
    frame_shape: tuple[int, int],
) -> Optional[tuple[int, int, int, int]]:
    x1, y1 = start
    x2, y2 = end
    frame_h, frame_w = frame_shape[:2]
    x1 = int(np.clip(x1, PANEL_WIDTH, frame_w - 1))
    x2 = int(np.clip(x2, PANEL_WIDTH, frame_w - 1))
    y1 = int(np.clip(y1, 0, frame_h - 1))
    y2 = int(np.clip(y2, 0, frame_h - 1))
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    if w <= 20 or h <= 20:
        return None
    return x, y, w, h


def detection_at_point(detections: list[Detection], point: tuple[int, int]) -> Optional[int]:
    px, py = point
    for idx, det in enumerate(detections):
        if cv2.pointPolygonTest(det.contour, (float(px), float(py)), False) >= 0:
            return idx
    return None


def filter_demo_detections_by_roi(
    detections: list[Detection],
    roi: Optional[tuple[int, int, int, int]],
) -> list[Detection]:
    if roi is None:
        return detections
    x, y, w, h = roi
    filtered = []
    for det in detections:
        cx, cy = det.center_px
        if x <= cx <= x + w and y <= cy <= y + h:
            filtered.append(det)
    return filtered


def palette() -> list[tuple[int, int, int]]:
    return [
        (255, 119, 119),
        (94, 203, 255),
        (118, 224, 137),
        (255, 204, 92),
        (199, 146, 255),
        (255, 143, 213),
        (72, 217, 194),
        (255, 164, 92),
        (152, 191, 255),
        (190, 232, 88),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intel RealSense D457 mixed-case depalletizing vision pipeline"
    )
    parser.add_argument("--model", default="FastSAM-s.pt", help="FastSAM or YOLOv8-seg model path")
    parser.add_argument(
        "--segmentation-backend",
        choices=("auto", "fastsam", "yolo", "depth"),
        default="auto",
        help="Segmentation mode: auto/FastSAM/YOLO zero-shot AI, or depth-edge fallback",
    )
    parser.add_argument("--width", type=int, default=1280, help="Color/depth stream width")
    parser.add_argument("--height", type=int, default=720, help="Color/depth stream height")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS")
    parser.add_argument("--pallet-width-mm", type=float, default=DEFAULT_PALLET_WIDTH_MM, help="Pallet width across the camera horizontal FOV")
    parser.add_argument("--pallet-depth-mm", type=float, default=DEFAULT_PALLET_DEPTH_MM, help="Pallet depth across the camera vertical FOV")
    parser.add_argument("--imgsz", type=int, default=640, help="Segmentation inference size")
    parser.add_argument("--conf", type=float, default=0.35, help="Segmentation confidence threshold")
    parser.add_argument("--iou", type=float, default=0.70, help="Segmentation NMS IoU threshold")
    parser.add_argument("--min-area", type=int, default=2000, help="Minimum mask area in pixels")
    parser.add_argument("--tie-depth-mm", type=float, default=10.0, help="Height tie threshold in mm")
    parser.add_argument("--roi-file", type=Path, default=DEFAULT_ROI_FILE, help="ROI config JSON path")
    parser.add_argument("--device", default=None, help="Ultralytics device, for example '0', 'cuda:0', or 'cpu'")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on supported CUDA devices")
    parser.add_argument("--no-auto-roi-load", action="store_true", help="Do not auto-load saved ROI")
    parser.add_argument("--demo", action="store_true", help="Start with synthetic demo mode enabled")
    return parser.parse_args()


def run_operator_console(args: argparse.Namespace, recommendation: CameraMountRecommendation) -> int:
    state = {
        "demo_enabled": bool(args.demo),
        "connection_status": "demo feed" if args.demo else "connecting",
        "model": None,
        "model_type": "demo" if args.demo else "camera",
        "pipeline": None,
        "align": None,
        "depth_scale": None,
        "intrinsics": None,
        "roi": DEFAULT_DEMO_ROI if args.demo else None,
        "last_error": "",
        "capture_requested": False,
        "captured": False,
        "confirmed": False,
        "frozen_frame": None,
        "detections": [],
        "selected_idx": None,
        "dragging": False,
        "drag_start": None,
        "drag_current": None,
        "mouse_pos": None,
        "exit_requested": False,
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        point = (x, y)
        state["mouse_pos"] = point
        if event == cv2.EVENT_LBUTTONDOWN:
            if click_inside(point, DEMO_TOGGLE_RECT):
                state["demo_enabled"] = not state["demo_enabled"]
                reset_capture_state(state)
                state["roi"] = DEFAULT_DEMO_ROI if state["demo_enabled"] else None
                state["connection_status"] = "demo feed" if state["demo_enabled"] else "camera idle"
                return
            if click_inside(point, CAPTURE_RECT):
                state["capture_requested"] = True
                return
            if click_inside(point, CONFIRM_RECT):
                confirm_selected_detection(state)
                return
            if click_inside(point, NEXT_RECT):
                cycle_selected_detection(state)
                return
            if click_inside(point, REDRAG_RECT):
                reset_capture_state(state, keep_roi=True)
                return
            if click_inside(point, EXIT_RECT):
                state["exit_requested"] = True
                return
            if state["captured"]:
                selected = detection_at_point(state["detections"], point)
                if selected is not None:
                    state["selected_idx"] = selected
                    state["confirmed"] = False
                return
            if x > PANEL_WIDTH:
                state["dragging"] = True
                state["drag_start"] = point
                state["drag_current"] = point
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["drag_current"] = point
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            rect = normalize_rect(state["drag_start"], point, (720, 1280))
            if rect is not None:
                state["roi"] = rect
                reset_capture_state(state, keep_roi=True)
            state["drag_start"] = None
            state["drag_current"] = None

    cv2.namedWindow("Pallet Sight Operator Console", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pallet Sight Operator Console", 1280, 720)
    cv2.setMouseCallback("Pallet Sight Operator Console", on_mouse)

    start = time.perf_counter()
    last_time = time.perf_counter()
    fps_smooth = 0.0

    while True:
        now = time.perf_counter()
        instantaneous_fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        fps_smooth = instantaneous_fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * instantaneous_fps)

        if state["capture_requested"]:
            capture_once(args, state, now - start)
            state["capture_requested"] = False

        if state["exit_requested"]:
            break

        if state["captured"] and state["frozen_frame"] is not None:
            color = state["frozen_frame"].copy()
            detections = state["detections"]
            best_idx = state["selected_idx"]
            roi = state["roi"]
            model_type = str(state["model_type"])
        elif state["demo_enabled"]:
            stop_real_resources(state)
            color, _preview_detections, _preview_best = synthetic_demo_frame(now - start)
            detections = []
            best_idx = None
            roi = state["roi"]
            model_type = "demo"
        else:
            color, detections, best_idx, roi, model_type = live_camera_preview_or_status(args, state)

        drag_roi = None
        if state["dragging"] and state["drag_start"] and state["drag_current"]:
            drag_roi = normalize_rect(state["drag_start"], state["drag_current"], color.shape[:2])

        capture_state = capture_state_label(state)

        display = draw_operator_ui(
            frame=color,
            roi=roi,
            detections=detections,
            best_idx=best_idx,
            fps=fps_smooth,
            recommendation=recommendation,
            model_type=model_type,
            demo_enabled=bool(state["demo_enabled"]),
            connection_status=str(state["connection_status"]),
            capture_state=capture_state,
            drag_roi=drag_roi,
            mouse_pos=state["mouse_pos"],
        )
        cv2.imshow("Pallet Sight Operator Console", display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("d"):
            state["demo_enabled"] = not state["demo_enabled"]
            reset_capture_state(state)
            state["roi"] = DEFAULT_DEMO_ROI if state["demo_enabled"] else None
            state["connection_status"] = "demo feed" if state["demo_enabled"] else "camera idle"
        if key == ord("c") or key == 13:
            state["capture_requested"] = True
        if key == ord("n"):
            cycle_selected_detection(state)
        if key == ord("p"):
            confirm_selected_detection(state)
        if key == ord("r"):
            reset_capture_state(state, keep_roi=True)
        if key == ord("s") and state["roi"] is not None:
            save_roi(args.roi_file, state["roi"], color.shape[:2])
            print(f"Saved ROI to {args.roi_file}")
        if key == ord("l"):
            loaded_roi = load_roi(args.roi_file, color.shape[:2])
            if loaded_roi is not None:
                state["roi"] = loaded_roi
                reset_capture_state(state, keep_roi=True)
                print(f"Loaded ROI: {loaded_roi}")

    stop_real_resources(state)
    cv2.destroyAllWindows()
    return 0


def reset_capture_state(state: dict, keep_roi: bool = False) -> None:
    roi = state.get("roi") if keep_roi else None
    state["capture_requested"] = False
    state["captured"] = False
    state["confirmed"] = False
    state["frozen_frame"] = None
    state["detections"] = []
    state["selected_idx"] = None
    state["dragging"] = False
    state["drag_start"] = None
    state["drag_current"] = None
    if not keep_roi:
        state["roi"] = roi


def capture_state_label(state: dict) -> str:
    if state["confirmed"]:
        return "confirmed"
    if state["captured"]:
        return "review pick"
    if state["connection_status"] in {"disconnected", "fault"} and not state["demo_enabled"]:
        return "camera blocked"
    return "ready to capture"


def cycle_selected_detection(state: dict) -> None:
    detections = state.get("detections", [])
    if not detections:
        return
    current = state.get("selected_idx")
    state["selected_idx"] = 0 if current is None else (int(current) + 1) % len(detections)
    state["confirmed"] = False


def confirm_selected_detection(state: dict) -> None:
    idx = state.get("selected_idx")
    detections = state.get("detections", [])
    if idx is None or idx < 0 or idx >= len(detections):
        return
    state["confirmed"] = True
    det = detections[idx]
    camera = det.camera_xyz_mm if det.camera_xyz_mm is not None else np.zeros(3)
    robot = det.robot_xyz_mm if det.robot_xyz_mm is not None else np.zeros(3)
    print(
        "CONFIRMED PICK "
        f"box={idx + 1} pixel={det.center_px} "
        f"camera_mm=({camera[0]:.1f}, {camera[1]:.1f}, {camera[2]:.1f}) "
        f"robot_mm=({robot[0]:.1f}, {robot[1]:.1f}, {robot[2]:.1f})"
    )


def capture_once(args: argparse.Namespace, state: dict, elapsed_s: float) -> None:
    state["confirmed"] = False
    if state["demo_enabled"]:
        color, detections, best_idx = synthetic_demo_frame(elapsed_s)
        detections = filter_demo_detections_by_roi(detections, state["roi"])
        if detections:
            best_idx = choose_highest_detection(detections, args.tie_depth_mm, (640.0, 360.0))
        else:
            best_idx = None
        state["frozen_frame"] = color
        state["detections"] = detections
        state["selected_idx"] = best_idx
        state["captured"] = True
        state["connection_status"] = "demo feed"
        state["model_type"] = "demo"
        return

    ensure_real_resources(args, state)
    color, detections, best_idx, roi, model_type = real_frame_or_status(args, state)
    state["frozen_frame"] = color
    state["detections"] = detections
    state["selected_idx"] = best_idx
    state["roi"] = roi
    state["model_type"] = model_type
    state["captured"] = bool(detections)


def preview_frame_or_status(
    state: dict,
) -> tuple[np.ndarray, list[Detection], Optional[int], Optional[tuple[int, int, int, int]], str]:
    frame = np.full((720, 1280, 3), (34, 38, 44), dtype=np.uint8)
    cv2.rectangle(frame, (420, 210), (1160, 510), (45, 52, 62), -1)
    if state.get("connection_status") in {"disconnected", "fault"}:
        cv2.rectangle(frame, (420, 210), (1160, 510), (235, 101, 101), 2)
        put_text(frame, "Camera Mode Blocked", (460, 278), 0.95, (245, 248, 252), 2)
        put_text(frame, "RealSense / AI pipeline could not start.", (460, 330), 0.62, (190, 202, 216), 1)
        wrapped = wrap_text(state.get("last_error", "No status message."), 58)
        y = 376
        for line in wrapped[:3]:
            put_text(frame, line, (460, y), 0.52, (235, 190, 190), 1)
            y += 28
    else:
        cv2.rectangle(frame, (420, 210), (1160, 510), (82, 153, 255), 2)
        put_text(frame, "Camera Mode Idle", (460, 278), 0.95, (245, 248, 252), 2)
        put_text(frame, "No processing is running yet.", (460, 330), 0.62, (190, 202, 216), 1)
        put_text(frame, "Drag ROI if needed, then click CAPTURE once.", (460, 370), 0.52, (164, 176, 190), 1)
        put_text(frame, "On capture, the app connects to RealSense and runs segmentation.", (460, 404), 0.52, (164, 176, 190), 1)
    return frame, [], None, state.get("roi"), str(state["model_type"])


def live_camera_preview_or_status(
    args: argparse.Namespace,
    state: dict,
) -> tuple[np.ndarray, list[Detection], Optional[int], Optional[tuple[int, int, int, int]], str]:
    """Show aligned RGB preview before capture without running segmentation."""

    ensure_real_resources(args, state, need_model=False)
    if state["pipeline"] is None or state["align"] is None:
        return preview_frame_or_status(state)

    try:
        frames = state["pipeline"].wait_for_frames(timeout_ms=1000)
        aligned = state["align"].process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            raise RuntimeError("RealSense returned no RGB frame for live preview.")

        color = np.asanyarray(color_frame.get_data())
        if depth_frame:
            depth = np.asanyarray(depth_frame.get_data())
        else:
            depth = np.zeros(color.shape[:2], dtype=np.uint16)
        color_roi, _depth_roi, _roi_mask = apply_roi(color, depth, state["roi"])
        state["connection_status"] = "rgb live"
        state["last_error"] = ""
        return color_roi, [], None, state["roi"], "rgb-live"
    except Exception as exc:
        state["connection_status"] = "fault"
        state["last_error"] = str(exc)
        stop_real_resources(state)
        return preview_frame_or_status(state)


def ensure_real_resources(args: argparse.Namespace, state: dict, need_model: bool = True) -> None:
    model_ready = state["model"] is not None or state["model_type"] == "depth-edge"
    if state["pipeline"] is not None and (not need_model or model_ready):
        state["connection_status"] = "connected"
        return

    try:
        missing = []
        if need_model and args.segmentation_backend != "depth" and (FastSAM is None or YOLO is None):
            missing.append("ultralytics (`pip install ultralytics`)")
        if rs is None:
            missing.append("pyrealsense2 (`pip install pyrealsense2`)")
        if missing:
            raise RuntimeError("Missing dependencies: " + "; ".join(missing))

        if need_model and state["model"] is None and state["model_type"] != "depth-edge":
            model, model_type = load_segmentation_model(args.model, args.segmentation_backend)
            state["model"] = model
            state["model_type"] = model_type
        if need_model and args.segmentation_backend == "depth":
            state["model"] = None
            state["model_type"] = "depth-edge"

        if state["pipeline"] is None:
            pipeline, align = setup_realsense(args.width, args.height, args.fps)
            color_image, _depth_image, color_profile, depth_scale = warmup_and_get_frame(
                pipeline, align
            )
            state["pipeline"] = pipeline
            state["align"] = align
            state["depth_scale"] = depth_scale
            state["intrinsics"] = color_profile.get_intrinsics()
            if not args.no_auto_roi_load:
                state["roi"] = load_roi(args.roi_file, color_image.shape[:2])
            state["connection_status"] = "connected"
    except Exception as exc:
        state["connection_status"] = "disconnected"
        state["last_error"] = str(exc)
        stop_real_resources(state)


def stop_real_resources(state: dict) -> None:
    pipeline = state.get("pipeline")
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception:
            pass
    state["pipeline"] = None
    state["align"] = None
    state["depth_scale"] = None
    state["intrinsics"] = None


def real_frame_or_status(
    args: argparse.Namespace,
    state: dict,
) -> tuple[np.ndarray, list[Detection], Optional[int], Optional[tuple[int, int, int, int]], str]:
    if state["pipeline"] is None or (state["model"] is None and state["model_type"] != "depth-edge"):
        return status_frame(state["last_error"]), [], None, None, str(state["model_type"])

    try:
        frames = state["pipeline"].wait_for_frames(timeout_ms=5000)
        aligned = state["align"].process(frames)
        aligned_depth = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not aligned_depth or not color_frame:
            raise RuntimeError("RealSense returned incomplete aligned frames.")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(aligned_depth.get_data())
        color_roi, depth_roi, roi_mask = apply_roi(color, depth, state["roi"])
        if state["model_type"] == "depth-edge":
            masks = segment_boxes_from_depth(
                depth_raw=depth_roi,
                roi_mask=roi_mask,
                depth_scale=float(state["depth_scale"]),
                min_area_px=args.min_area,
            )
        else:
            masks = segment_boxes(
                model=state["model"],
                model_type=str(state["model_type"]),
                color_bgr=color_roi,
                roi_mask=roi_mask,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                min_area_px=args.min_area,
                device=args.device,
                half=args.half,
            )
        frame_center = (color.shape[1] / 2.0, color.shape[0] / 2.0)
        detections, best_idx = build_detections(
            masks=masks,
            depth_raw=depth_roi,
            depth_scale=float(state["depth_scale"]),
            intrinsics=state["intrinsics"],
            tie_depth_mm=args.tie_depth_mm,
            frame_center=frame_center,
        )
        state["connection_status"] = "connected"
        state["last_error"] = ""
        return color_roi, detections, best_idx, state["roi"], str(state["model_type"])
    except Exception as exc:
        state["connection_status"] = "fault"
        state["last_error"] = str(exc)
        stop_real_resources(state)
        return status_frame(state["last_error"]), [], None, None, str(state["model_type"])


def status_frame(message: str) -> np.ndarray:
    frame = np.full((720, 1280, 3), (34, 38, 44), dtype=np.uint8)
    cv2.rectangle(frame, (420, 210), (1160, 510), (45, 52, 62), -1)
    cv2.rectangle(frame, (420, 210), (1160, 510), (235, 101, 101), 2)
    put_text(frame, "Camera Mode", (460, 270), 0.95, (245, 248, 252), 2)
    put_text(frame, "RealSense / AI pipeline is not ready.", (460, 318), 0.62, (190, 202, 216), 1)
    wrapped = wrap_text(message or "No status message.", 58)
    y = 370
    for line in wrapped[:4]:
        put_text(frame, line, (460, y), 0.52, (235, 190, 190), 1)
        y += 28
    put_text(frame, "Click DEMO OFF/ON to retry or return to demo.", (460, 474), 0.52, (164, 176, 190), 1)
    return frame


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def synthetic_demo_frame(t: float) -> tuple[np.ndarray, list[Detection], Optional[int]]:
    frame = np.full((720, 1280, 3), (42, 45, 48), dtype=np.uint8)
    cv2.rectangle(frame, (390, 70), (1210, 680), (84, 72, 54), -1)
    cv2.rectangle(frame, (410, 90), (1190, 660), (126, 105, 73), 3)

    boxes = [
        ((440, 125, 250, 155), 845.0),
        ((700, 125, 210, 155), 780.0),
        ((920, 125, 245, 155), 910.0),
        ((440, 295, 195, 185), 830.0),
        ((650, 295, 255, 185), 765.0),
        ((920, 295, 245, 185), 835.0),
        ((440, 495, 275, 140), 900.0),
        ((730, 495, 190, 140), 820.0),
        ((935, 495, 230, 140), 875.0),
    ]
    detections: list[Detection] = []
    pulse = 18.0 * math.sin(t * 1.5)
    for idx, ((x, y, w, h), depth_mm) in enumerate(boxes):
        shade = 118 + (idx % 3) * 18
        cv2.rectangle(frame, (x, y), (x + w, y + h), (shade, shade + 18, shade + 35), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (76, 66, 51), 2)
        cv2.line(frame, (x + 12, y + 14), (x + w - 12, y + 14), (150, 139, 111), 1)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        contour = np.array(
            [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]],
            dtype=np.int32,
        )
        cv2.drawContours(mask, [contour], -1, 255, -1)
        center_px = (x + w // 2, y + h // 2)
        camera_xyz = np.array(
            [float(center_px[0] - 640), float(center_px[1] - 360), depth_mm + (pulse if idx == 4 else 0.0)],
            dtype=np.float64,
        )
        detections.append(
            Detection(
                mask=mask,
                contour=contour,
                center_px=center_px,
                avg_depth_m=camera_xyz[2] / 1000.0,
                area_px=float(w * h),
                camera_xyz_mm=camera_xyz,
                robot_xyz_mm=transform_camera_to_robot(camera_xyz),
            )
        )

    best_idx = choose_highest_detection(detections, 10.0, (640.0, 360.0))
    return frame, detections, best_idx


def main() -> int:
    args = parse_args()
    recommendation = calculate_mounting_height(
        pallet_width_mm=args.pallet_width_mm,
        pallet_depth_mm=args.pallet_depth_mm,
    )
    print_mounting_recommendation(recommendation, args.pallet_width_mm, args.pallet_depth_mm)

    print("Launching operator console. Click DEMO ON/OFF or press d to toggle.")
    return run_operator_console(args, recommendation)


if __name__ == "__main__":
    raise SystemExit(main())
