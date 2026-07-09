# Mix Case Palletizing - D457 Depalletizing Vision

Production-oriented Python/OpenCV operator console for a top-down Intel RealSense
D457 depalletizing station.

## Features

- D457 mounting-height recommendation from configurable pallet width/depth.
- Live RGB preview before capture, with simulated or real processing after capture.
- Drag ROI selection directly in the live/demo view.
- One-shot capture cycle: capture once, freeze result, review candidate, confirm.
- FastSAM/YOLO zero-shot segmentation path for mixed cardboard boxes.
- Optional depth-edge segmentation fallback with no teaching/training.
- RealSense depth-to-color alignment with `rs.align(rs.stream.color)`.
- Highest-box selection by smallest Z distance, with center-of-FOV tie-break.
- 2D pixel center to 3D camera coordinates via RealSense deprojection.
- Placeholder camera-to-robot homogeneous transform with calibration notes.

## Install

```powershell
pip install -r requirements.txt
```

The first run may download the selected Ultralytics model, for example
`FastSAM-s.pt`.

## Run

Demo mode:

```powershell
python depalletizing_realsense_d457.py --demo
```

Camera mode:

```powershell
python depalletizing_realsense_d457.py --model FastSAM-s.pt --device 0 --half --pallet-width-mm 1100 --pallet-depth-mm 1100
```

Use `--device cpu` if no CUDA GPU is available.

Web UI with real D457 depth capture:

```powershell
python realsense_web_bridge.py --segmentation-backend depth
```

Then open the web UI, turn **Demo result** off so the status shows
**RealSense depth mode**, drag the ROI, and press **Capture**. The browser sends
preview/capture requests to the local bridge at `http://127.0.0.1:8765`; the
bridge streams the RealSense RGB preview, captures aligned D457 color/depth, and
returns measured Z, box length/width, and camera/robot XYZ.

The bridge defaults to `640x480@30` because it works on USB 2.x D400 devices.
For a D457/GMSL or USB 3 connection, you can request a higher profile, for
example `--width 1280 --height 720 --fps 30` if the camera reports support.
Length/width output applies `--dimension-scale` to compensate for masks that
land slightly inside the true carton edge. The current default is calibrated
from a measured 180 x 95 mm reference box; adjust it during commissioning if
your measured reference box differs.

Segmentation choices:

```powershell
# Default: infer FastSAM or YOLO from the model file
python depalletizing_realsense_d457.py --segmentation-backend auto

# Explicit zero-shot AI options, no custom teaching dataset required
python depalletizing_realsense_d457.py --segmentation-backend fastsam --model FastSAM-s.pt
python depalletizing_realsense_d457.py --segmentation-backend yolo --model yolov8n-seg.pt

# Depth-driven fallback, useful for scenes with clear height discontinuities
python depalletizing_realsense_d457.py --segmentation-backend depth
```

## Operator Flow

1. Drag the pallet ROI.
2. Click **CAPTURE** to process one frame only.
3. Review the proposed pick candidate.
4. Click another box or **NEXT** to override the selection.
5. Click **CONFIRM** to accept the pick coordinates.
6. Click **REDRAG ROI** to exit the review state and define a new ROI.

## Real Mode Notes

Camera mode shows live aligned RGB before **CAPTURE** so the operator can verify
camera angle and drag the ROI on the real image. On capture, the app masks
outside the ROI, runs the selected segmentation backend, computes per-box depth,
computes box length/width from RealSense intrinsics, rejects masks without valid
depth, and proposes the highest box.

If camera mode reports blocked, check D457 power, cable/GMSL adapter, firmware,
and whether another process owns the camera.

## Where Processing Runs

The React web demo is an operator-interface prototype. In demo mode, before
capture it can show the browser RGB camera preview when camera permission is
available. When **Demo result on** is enabled, pressing **Capture** switches from
live RGB to a simulated detection result. When **RealSense depth mode** is
enabled, preview and capture both come from `realsense_web_bridge.py`, which owns
the D457 stream, performs `rs.align(rs.stream.color)`, masks outside the ROI,
segments candidates, and returns measured Z and box length/width in millimeters.
In depth mode, shadows are rejected because candidates must be real depth
foreground surfaces with valid depth, not only RGB contrast. A browser cannot directly run
`pyrealsense2`, align D457 depth to RGB, or call the RealSense SDK without this
local bridge.

The real depalletizing processing runs in `depalletizing_realsense_d457.py` on
the Windows PC connected to the RealSense camera. That Python process owns the
D457 stream, performs `rs.align(rs.stream.color)`, applies the ROI mask, runs
FastSAM/YOLO zero-shot segmentation or depth-edge segmentation, computes depth
and 3D camera coordinates, applies the hand-eye transform, and prints/returns
the robot target. For deployment, run the Python process as the vision
service/operator console, or run `realsense_web_bridge.py` for the web UI.

## Hand-Eye Matrix Example

The app uses an eye-to-hand transform:

```text
P_robot = T_robot_camera * P_camera
```

Where `P_camera = [Xc, Yc, Zc, 1]^T` in millimeters and
`T_robot_camera` is a 4x4 homogeneous matrix. Example:

```text
T_robot_camera =
[  1.000   0.000   0.000   500.0 ]
[  0.000  -1.000   0.000   250.0 ]
[  0.000   0.000  -1.000  1200.0 ]
[  0.000   0.000   0.000     1.0 ]
```

For a RealSense pick point `P_camera = [137, 27, 756, 1]^T`:

```text
Xr =  1*137 + 0*27  + 0*756 + 500  = 637 mm
Yr =  0*137 - 1*27  + 0*756 + 250  = 223 mm
Zr =  0*137 + 0*27  - 1*756 + 1200 = 444 mm
```

So the robot target is approximately `[637, 223, 444]` mm before adding tool,
gripper, and pick-approach offsets.

To calibrate the real matrix:

1. Rigidly mount the D457 and do not move it afterward.
2. Collect at least 8 to 15 corresponding 3D points visible to both systems.
   For each calibration point, record the camera-space point from RealSense and
   the same physical point in robot-base coordinates.
3. Spread points across the pallet: corners, center, edges, and multiple box
   heights. Do not use only one flat plane.
4. Solve the rigid transform with SVD/Kabsch or OpenCV calibration tooling.
5. Check axis direction. A top-down camera often has camera +Z pointing downward
   away from the camera, while robot +Z usually points upward. This is why the
   example matrix flips Y and Z.
6. Validate with independent points. Keep residual error below the tolerance
   required by the gripper and box placement.

## Industrial Deployment Notes

- Treat the ROI as a hard safety/filtering boundary. RGB and depth outside the
  ROI are masked before segmentation, and generated masks are intersected with
  the ROI before depth and robot coordinates are computed.
- A capture with zero in-ROI detections must not create a robot pick. Redraw the
  ROI, inspect pallet placement, or clear camera/lighting faults before retrying.
- Keep the production flow one-shot: capture, freeze, review, optionally override
  to another in-ROI box, then confirm. Do not stream robot targets continuously.
- Save a validated ROI for the cell after commissioning and re-check it after
  camera mount, pallet guide, conveyor, or lighting changes.
