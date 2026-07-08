# D457 Depalletizing Operator Interface Design

## Product Design Brief

Build a simple, modern industrial operator console for a RealSense D457
top-down depalletizing cell. The screen should prioritize live confidence:
what the camera sees, which boxes were segmented, which box will be picked
next, and the exact camera/robot coordinates.

## Screen Layout

- Main canvas: full live RGB camera stream with background outside the pallet
  ROI dimmed to charcoal.
- Pallet ROI: thin cyan rectangle labeled `PALLET ROI`.
- Box masks: distinct high-contrast contour colors, one label per box with
  depth in millimeters.
- Highest box: thicker contour, yellow crosshair, circular pick marker.
- Left status rail: translucent dark panel, approximately 380 px wide.

## Status Rail Content

- Header: `D457 Depalletizing`
- Subtitle: `Mixed case top-down vision`
- Chips: `LIVE`, model type, current FPS.
- Setup metrics: ROI state, detection count, minimum mount Z, recommended Z.
- Pick target: pixel center, camera XYZ, robot XYZ.
- Footer controls: `r ROI`, `s save`, `l load`, `q quit`.

## Visual System

- Background panel: deep neutral charcoal for low glare on factory monitors.
- Accent colors: cyan for ROI, green for live state, blue for model state,
  amber for FPS and selected pick marker.
- Typography: compact OpenCV sans-serif rendering for portability. Use a
  medium-weight industrial UI font such as Inter or DIN if ported to Qt/Web.
- Density: information should fit in one view at 1280 x 720 without scrolling.

## UX States

- Setup state: ROI picker shown first unless a saved ROI is loaded.
- Running state: live detections and highest-box decision visible every frame.
- Empty state: `No valid pick target` when segmentation/depth is insufficient.
- Camera fault: console prints a clear disconnection message and exits safely.
- Rework state: operator can press `r` any time to redraw pallet ROI.

## Figma Handoff Note

The Figma connector was available, but this session did not expose the account
`whoami` plan discovery tool required before creating a new Figma file. This
spec mirrors the interface already implemented in `depalletizing_realsense_d457.py`
and can be used as the source brief for a future Figma file once a plan key or
target Figma file is provided.
