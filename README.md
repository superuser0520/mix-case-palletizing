# Mix Case Palletizing - D457 Depalletizing Vision

Production-oriented Python/OpenCV operator console for a top-down Intel RealSense
D457 depalletizing station.

## Features

- D457 mounting-height recommendation for an 1100 mm x 1100 mm pallet.
- Demo/camera toggle with guided operator workflow.
- Drag ROI selection directly in the live/demo view.
- One-shot capture cycle: capture once, freeze result, review candidate, confirm.
- FastSAM/YOLO segmentation path for mixed cardboard boxes.
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
python depalletizing_realsense_d457.py --model FastSAM-s.pt --device 0 --half
```

Use `--device cpu` if no CUDA GPU is available.

## Operator Flow

1. Drag the pallet ROI.
2. Click **CAPTURE** to process one frame only.
3. Review the proposed pick candidate.
4. Click another box or **NEXT** to override the selection.
5. Click **CONFIRM** to accept the pick coordinates.
6. Click **REDRAG ROI** to exit the review state and define a new ROI.

## Real Mode Notes

Camera mode idles until **CAPTURE** is pressed. On capture, the app starts the
RealSense pipeline, aligns depth to color, runs segmentation, computes per-box
depth, and proposes the highest box.

If camera mode reports blocked, check D457 power, cable/GMSL adapter, firmware,
and whether another process owns the camera.

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
