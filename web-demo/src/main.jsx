import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  Box,
  Camera,
  Check,
  ChevronRight,
  Crosshair,
  Maximize2,
  MousePointer2,
  RefreshCw,
  ScanLine,
  ToggleLeft,
  Zap,
} from 'lucide-react';
import './styles.css';

const boxes = [
  { id: 1, x: 8, y: 9, w: 28, h: 21, z: 846, lengthMm: 310, widthMm: 230, color: '#B99EF8' },
  { id: 2, x: 38, y: 9, w: 24, h: 21, z: 781, lengthMm: 265, widthMm: 230, color: '#F2C96B' },
  { id: 3, x: 64, y: 9, w: 28, h: 21, z: 910, lengthMm: 310, widthMm: 230, color: '#83A633' },
  { id: 4, x: 8, y: 34, w: 23, h: 25, z: 832, lengthMm: 275, widthMm: 250, color: '#69BFE8' },
  { id: 5, x: 33, y: 34, w: 29, h: 25, z: 756, lengthMm: 320, widthMm: 250, color: '#DFA1D8' },
  { id: 6, x: 64, y: 34, w: 28, h: 25, z: 836, lengthMm: 310, widthMm: 250, color: '#B99EF8' },
  { id: 7, x: 8, y: 63, w: 31, h: 19, z: 901, lengthMm: 340, widthMm: 210, color: '#DEF4A5' },
  { id: 8, x: 41, y: 63, w: 23, h: 19, z: 819, lengthMm: 255, widthMm: 210, color: '#6AA9F2' },
  { id: 9, x: 66, y: 63, w: 26, h: 19, z: 874, lengthMm: 285, widthMm: 210, color: '#F6A06E' },
];

const D457_DEPTH_FOV = { h: 87, v: 58, tolerance: 3 };
const D457_MIN_RANGE_MM = 600;
const MOUNT_MARGIN = 0.15;
const DEFAULT_PALLET = { width: '1100', depth: '1100' };
const REALSENSE_BRIDGE_URL = 'http://127.0.0.1:8765/api/capture';
const REALSENSE_PREVIEW_URL = 'http://127.0.0.1:8765/api/preview';

function isBoxInsideRoi(item, roi) {
  const centerX = item.x + item.w / 2;
  const centerY = item.y + item.h / 2;
  return centerX >= roi.x && centerX <= roi.x + roi.w && centerY >= roi.y && centerY <= roi.y + roi.h;
}

function chooseHighest(items) {
  if (!items.length) return null;
  return items.reduce((best, item) => {
    if (item.z !== best.z) return item.z < best.z ? item : best;
    const itemCenter = Math.hypot(item.x + item.w / 2 - 50, item.y + item.h / 2 - 50);
    const bestCenter = Math.hypot(best.x + best.w / 2 - 50, best.y + best.h / 2 - 50);
    return itemCenter < bestCenter ? item : best;
  }, items[0]);
}

function parseDimension(value, fallback) {
  const parsed = Number(String(value).trim());
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function calculateMountingHeight(pallet) {
  const conservativeH = D457_DEPTH_FOV.h - D457_DEPTH_FOV.tolerance;
  const conservativeV = D457_DEPTH_FOV.v - D457_DEPTH_FOV.tolerance;
  const palletWidth = parseDimension(pallet.width, Number(DEFAULT_PALLET.width));
  const palletDepth = parseDimension(pallet.depth, Number(DEFAULT_PALLET.depth));
  const requiredByWidth = (palletWidth / 2) / Math.tan((conservativeH / 2) * Math.PI / 180);
  const requiredByDepth = (palletDepth / 2) / Math.tan((conservativeV / 2) * Math.PI / 180);
  const minHeight = Math.max(requiredByWidth, requiredByDepth, D457_MIN_RANGE_MM);
  return {
    minHeight: Math.round(minHeight),
    recommendedHeight: Math.round(minHeight * (1 + MOUNT_MARGIN)),
    conservativeH,
    conservativeV,
    palletWidth,
    palletDepth,
  };
}

function App() {
  const [view, setView] = useState('vision');
  const [demoMode, setDemoMode] = useState(true);
  const [cameraStatus, setCameraStatus] = useState('Demo mode');
  const [captured, setCaptured] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [roi, setRoi] = useState({ x: 6, y: 6, w: 88, h: 88 });
  const [pallet, setPallet] = useState(DEFAULT_PALLET);
  const [cameraStream, setCameraStream] = useState(null);
  const [actualBoxes, setActualBoxes] = useState([]);
  const [capturedFrame, setCapturedFrame] = useState('');
  const [bridgePreviewFrame, setBridgePreviewFrame] = useState('');
  const cameraStreamRef = useRef(null);
  const videoElementRef = useRef(null);

  const mount = useMemo(() => calculateMountingHeight(pallet), [pallet]);
  const demoDetectedBoxes = useMemo(() => boxes.filter((item) => isBoxInsideRoi(item, roi)), [roi]);
  const detectedBoxes = demoMode ? demoDetectedBoxes : actualBoxes;
  const highestBox = useMemo(() => chooseHighest(detectedBoxes), [detectedBoxes]);
  const selectedBox = detectedBoxes.find((item) => item.id === selectedId) ?? null;
  const candidate = selectedBox ?? (captured ? highestBox : null);

  function stopCameraPreview() {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    videoElementRef.current = null;
    setCameraStream(null);
  }

  function startCameraPreview() {
    if (cameraStreamRef.current) return () => {};
    let active = true;
    const cameraTimeout = window.setTimeout(() => {
      if (active && !cameraStreamRef.current) setCameraStatus('Waiting for camera permission');
    }, 3500);

    setCameraStatus('Requesting RGB preview');
    if (!navigator.mediaDevices?.getUserMedia) {
      window.clearTimeout(cameraTimeout);
      setCameraStatus('Browser camera API unavailable');
      return () => {};
    }

    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      .then((stream) => {
        window.clearTimeout(cameraTimeout);
        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        cameraStreamRef.current = stream;
        setCameraStream(stream);
        setCameraStatus('RGB preview live');
      })
      .catch(() => {
        window.clearTimeout(cameraTimeout);
        if (active) setCameraStatus('No browser camera permission');
      });

    return () => {
      active = false;
      window.clearTimeout(cameraTimeout);
    };
  }

  useEffect(() => {
    const cancelPreviewRequest = demoMode ? startCameraPreview() : undefined;
    return () => {
      cancelPreviewRequest?.();
      stopCameraPreview();
    };
  }, [demoMode]);

  useEffect(() => {
    if (demoMode || captured) {
      setBridgePreviewFrame('');
      return undefined;
    }

    let active = true;
    let timer = 0;
    stopCameraPreview();
    setCameraStatus('Requesting RealSense preview');

    async function refreshRealSensePreview() {
      try {
        const response = await fetch(REALSENSE_PREVIEW_URL);
        const result = await response.json();
        if (!response.ok || !result.ok) {
          throw new Error(result.message || `RealSense preview returned ${response.status}`);
        }
        if (!active) return;
        setBridgePreviewFrame(result.frame ?? '');
        setCameraStatus(result.message || 'RealSense preview live');
      } catch (error) {
        if (!active) return;
        setBridgePreviewFrame('');
        setCameraStatus(`RealSense preview unavailable: ${error.message}`);
      } finally {
        if (active) {
          timer = window.setTimeout(refreshRealSensePreview, 550);
        }
      }
    }

    refreshRealSensePreview();
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [demoMode, captured]);

  function resetPick() {
    setCaptured(false);
    setConfirmed(false);
    setSelectedId(null);
    setActualBoxes([]);
    setCapturedFrame('');
    setBridgePreviewFrame('');
    if (demoMode) {
      window.setTimeout(() => startCameraPreview(), 60);
    }
  }

  async function capture() {
    setConfirmed(false);
    if (demoMode) {
      setCaptured(true);
      setActualBoxes([]);
      setCapturedFrame('');
      setSelectedId(chooseHighest(demoDetectedBoxes)?.id ?? null);
      return;
    }

    setCaptured(true);
    setActualBoxes([]);
    setCapturedFrame('');
    setBridgePreviewFrame('');
    setSelectedId(null);
    setCameraStatus('Capturing RealSense depth');

    try {
      const response = await fetch(REALSENSE_BRIDGE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.message || `RealSense bridge returned ${response.status}`);
      }
      const realBoxes = result.detections ?? [];
      setActualBoxes(realBoxes);
      setCapturedFrame(result.frame ?? '');
      setCameraStatus(result.message || `RealSense depth detected ${realBoxes.length} box(es)`);
      setSelectedId(result.bestId ?? chooseHighest(realBoxes)?.id ?? null);
    } catch (error) {
      setActualBoxes([]);
      setCapturedFrame('');
      setCameraStatus(`RealSense depth unavailable: ${error.message}`);
    }
  }

  function nextCandidate() {
    if (!captured || !detectedBoxes.length) return;
    const currentIndex = detectedBoxes.findIndex((item) => item.id === (candidate?.id ?? highestBox?.id));
    setSelectedId(detectedBoxes[(currentIndex + 1) % detectedBoxes.length].id);
    setConfirmed(false);
  }

  return (
    <main className="app-shell">
      <aside className="side-rail">
        <div className="brand">
          <div className="brand-mark"><ScanLine size={22} /></div>
          <div>
            <h1>Pallet Sight</h1>
            <p>One-shot depalletizing vision</p>
          </div>
        </div>

        <nav className="nav-list">
          <NavButton active={view === 'vision'} onClick={() => setView('vision')} icon={<Camera size={18} />} label="Vision" />
          <NavButton active={view === 'queue'} onClick={() => setView('queue')} icon={<Box size={18} />} label="Pick Queue" />
          <NavButton active={view === 'calibration'} onClick={() => setView('calibration')} icon={<Activity size={18} />} label="Calibration" />
        </nav>

        <section className="soft-card sequence-card">
          <p className="label">Walkthrough</p>
          <ol>
            <li>Drag pallet ROI</li>
            <li>Capture one frame</li>
            <li>Review before confirm</li>
          </ol>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h2>{view === 'vision' ? 'Depalletizing console' : view === 'queue' ? 'Pick queue' : 'Calibration setup'}</h2>
            <p>{view === 'vision' ? 'Capture once, evaluate the stack, then confirm the robot pick.' : view === 'queue' ? 'Confirmed picks and pending handoffs for the robot cell.' : 'Eye-to-hand transform, mount height, and ROI setup status.'}</p>
          </div>
          <div className="top-actions">
            <StatusPill label={cameraStatus} />
            <button className="toggle" onClick={() => { setDemoMode(!demoMode); resetPick(); }}>
              <ToggleLeft size={20} />
              {demoMode ? 'Demo result on' : 'RealSense depth mode'}
            </button>
          </div>
        </header>

        {view === 'vision' && (
          <>
            <div className="grid">
              <section className="vision-card">
                <div className="card-head">
                  <div>
                    <p className="label">ROI</p>
                    <h3>{captured ? 'Review pick' : 'Ready to capture'}</h3>
                  </div>
                  <button className="icon-button" title="Expand preview"><Maximize2 size={18} /></button>
                </div>
                <PalletPreview
                  captured={captured}
                  selectedId={candidate?.id ?? null}
                  detectedBoxes={detectedBoxes}
                  demoMode={demoMode}
                  cameraStream={cameraStream}
                  cameraStatus={cameraStatus}
                  bridgePreviewFrame={bridgePreviewFrame}
                  videoElementRef={videoElementRef}
                  capturedFrame={capturedFrame}
                  roi={roi}
                  setRoi={(next) => {
                    setRoi(next);
                    resetPick();
                  }}
                  onSelect={(id) => {
                    if (!captured) return;
                    setSelectedId(id);
                    setConfirmed(false);
                  }}
                />
              </section>

              <CandidatePanel
                captured={captured}
                confirmed={confirmed}
                candidate={candidate}
                detectedCount={detectedBoxes.length}
                modeLabel={demoMode ? 'Simulated depth' : 'RealSense depth'}
                onCapture={capture}
                onNext={nextCandidate}
                onConfirm={() => captured && candidate && setConfirmed(true)}
                onRedrag={resetPick}
              />
            </div>

            <Metrics captured={captured} detectedCount={detectedBoxes.length} mount={mount} demoMode={demoMode} />
          </>
        )}

        {view === 'queue' && <PickQueue candidate={candidate} confirmed={confirmed} />}
        {view === 'calibration' && <Calibration roi={roi} pallet={pallet} setPallet={setPallet} mount={mount} />}
      </section>
    </main>
  );
}

function NavButton({ active, icon, label, onClick }) {
  return (
    <button className={`nav-item ${active ? 'active' : ''}`} onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}

function CandidatePanel({ captured, confirmed, candidate, detectedCount, modeLabel, onCapture, onNext, onConfirm, onRedrag }) {
  const emptyCapture = captured && !candidate;
  const cameraXyz = candidate?.camera_xyz ?? [Math.round((candidate?.x ?? 0) + (candidate?.w ?? 0) / 2), Math.round((candidate?.y ?? 0) + (candidate?.h ?? 0) / 2), candidate?.z];
  const robotXyz = candidate?.robot_xyz ?? cameraXyz;
  const formatXyz = (values) => values?.every((value) => Number.isFinite(Number(value))) ? values.map((value) => Math.round(Number(value))).join(', ') : '--';
  const dimensions = candidate?.lengthMm && candidate?.widthMm ? `${candidate.lengthMm} x ${candidate.widthMm} mm` : '--';
  return (
    <aside className="candidate-panel">
      <div className="candidate-top">
        <p className="label">Pick Candidate</p>
        <h3>{candidate ? `Box ${candidate.id}` : emptyCapture ? 'No box in ROI' : 'No capture yet'}</h3>
        <p>{candidate ? 'Highest in-ROI box proposed. Operator may override.' : emptyCapture ? 'Redrag ROI or inspect pallet placement before confirming.' : 'Click Capture to evaluate the pallet ROI.'}</p>
      </div>

      <div className="ring-card">
        <div className="ring"><Crosshair size={32} /></div>
        <div>
          <span>{captured ? `${detectedCount} in ROI` : modeLabel}</span>
          <strong>{candidate ? `${candidate.z} mm` : '--'}</strong>
        </div>
      </div>

      <div className="coords">
        <Coord label="Length x Width" value={candidate ? dimensions : '--'} />
        <Coord label="Camera XYZ" value={candidate ? formatXyz(cameraXyz) : '--'} />
        <Coord label="Robot XYZ" value={candidate ? formatXyz(robotXyz) : '--'} />
      </div>

      <div className="actions">
        <button className="primary" onClick={onCapture}><MousePointer2 size={18} />Capture</button>
        <button className="secondary" disabled={!captured || detectedCount < 2} onClick={onNext}><ChevronRight size={18} />Next</button>
        <button className="confirm" disabled={!captured || !candidate} onClick={onConfirm}><Check size={18} />Confirm</button>
        <button className="ghost" onClick={onRedrag}><RefreshCw size={18} />Redrag ROI</button>
      </div>

      {confirmed && candidate && (
        <div className="confirmed">
          <Zap size={18} />
          Pick confirmed for Box {candidate.id}. Robot handoff ready.
        </div>
      )}
    </aside>
  );
}

function PalletPreview({ captured, selectedId, detectedBoxes, onSelect, demoMode, cameraStream, cameraStatus, bridgePreviewFrame, videoElementRef, capturedFrame, roi, setRoi }) {
  const stageRef = useRef(null);
  const videoRef = useRef(null);
  const dragStart = useRef(null);
  const [draftRoi, setDraftRoi] = useState(null);

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
      videoElementRef.current = videoRef.current;
    }
  }, [cameraStream, videoElementRef]);

  function pointFromEvent(event) {
    const rect = stageRef.current.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    return {
      x: Math.max(0, Math.min(100, x)),
      y: Math.max(0, Math.min(100, y)),
    };
  }

  function startDrag(event) {
    if (captured) return;
    if (event.currentTarget.setPointerCapture) {
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    const point = pointFromEvent(event);
    dragStart.current = point;
    setDraftRoi({ x: point.x, y: point.y, w: 0, h: 0 });
  }

  function moveDrag(event) {
    if (!dragStart.current || captured) return;
    const point = pointFromEvent(event);
    const x = Math.min(dragStart.current.x, point.x);
    const y = Math.min(dragStart.current.y, point.y);
    const w = Math.abs(point.x - dragStart.current.x);
    const h = Math.abs(point.y - dragStart.current.y);
    setDraftRoi({ x, y, w, h });
  }

  function finishDrag(event) {
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (draftRoi && draftRoi.w > 8 && draftRoi.h > 8) {
      setRoi(draftRoi);
    }
    dragStart.current = null;
    setDraftRoi(null);
  }

  const shownRoi = draftRoi ?? roi;
  const outlineFor = (item) => {
    if (Array.isArray(item.outline) && item.outline.length >= 3) {
      return item.outline;
    }
    return [
      { x: item.x, y: item.y },
      { x: item.x + item.w, y: item.y },
      { x: item.x + item.w, y: item.y + item.h },
      { x: item.x, y: item.y + item.h },
    ];
  };
  const pointString = (points) => points.map((point) => `${point.x},${point.y}`).join(' ');

  return (
    <div className="pallet-stage">
      <div
        ref={stageRef}
        className={`roi-frame ${captured ? 'locked' : ''}`}
        onPointerDown={startDrag}
        onPointerMove={moveDrag}
        onPointerUp={finishDrag}
        onPointerCancel={finishDrag}
      >
        {!captured && demoMode && cameraStream && <video ref={videoRef} className="camera-feed" autoPlay muted playsInline />}
        {!captured && !demoMode && bridgePreviewFrame && <img className="captured-frame" src={bridgePreviewFrame} alt="RealSense preview frame" />}
        {!captured && ((demoMode && !cameraStream) || (!demoMode && !bridgePreviewFrame)) && <div className="camera-empty">{cameraStatus}. Use RealSense depth mode only when the local bridge is running.</div>}
        {captured && demoMode && <div className="simulation-surface" />}
        {captured && !demoMode && capturedFrame && <img className="captured-frame" src={capturedFrame} alt="Captured camera frame" />}
        {captured && !demoMode && !capturedFrame && <div className="camera-empty">{cameraStatus}</div>}

        <div className="roi-mask top" style={{ height: `${shownRoi.y}%` }} />
        <div className="roi-mask left" style={{ top: `${shownRoi.y}%`, width: `${shownRoi.x}%`, height: `${shownRoi.h}%` }} />
        <div className="roi-mask right" style={{ top: `${shownRoi.y}%`, left: `${shownRoi.x + shownRoi.w}%`, height: `${shownRoi.h}%` }} />
        <div className="roi-mask bottom" style={{ top: `${shownRoi.y + shownRoi.h}%` }} />

        <div
          className="roi-box"
          style={{
            left: `${shownRoi.x}%`,
            top: `${shownRoi.y}%`,
            width: `${shownRoi.w}%`,
            height: `${shownRoi.h}%`,
          }}
        />

        {captured && (
          <svg className="box-outline-layer" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Detected box outlines">
            {detectedBoxes.map((item) => {
              const active = item.id === selectedId;
              const outline = outlineFor(item);
              const labelX = item.x + item.w / 2;
              const labelY = Math.max(5, item.y - 2);
              const label = active
                ? `Box ${item.id} - ${item.z}mm${item.lengthMm && item.widthMm ? ` - ${item.lengthMm}x${item.widthMm}mm` : ''}`
                : `#${item.id}`;
              return (
                <g
                  key={item.id}
                  className={`box-outline ${demoMode ? 'demo' : 'actual'} ${active ? 'active' : ''}`}
                  style={{ '--box-color': item.color }}
                  onClick={() => onSelect(item.id)}
                >
                  <polygon points={pointString(outline)} />
                  <circle cx={labelX} cy={item.y + item.h / 2} r={active ? 0.95 : 0.7} />
                  <text x={labelX} y={labelY}>{label}</text>
                </g>
              );
            })}
          </svg>
        )}
      </div>
      {!captured && <div className="capture-hint">Outside the ROI is masked out. Capture only evaluates the drag zone.</div>}
    </div>
  );
}

function PickQueue({ candidate, confirmed }) {
  const rows = [
    { step: 'Current candidate', box: candidate ? `Box ${candidate.id}` : 'Waiting', z: candidate ? `${candidate.z} mm` : '--', status: candidate ? (confirmed ? 'Confirmed' : 'Review') : 'Empty' },
    { step: 'Next cycle', box: 'Pending capture', z: '--', status: 'Idle' },
    { step: 'Robot handoff', box: confirmed ? `Box ${candidate.id}` : 'No pick', z: confirmed ? `${candidate.z} mm` : '--', status: confirmed ? 'Ready' : 'Blocked' },
  ];
  return (
    <section className="table-card">
      <p className="label">Queue</p>
      <h3>Pick queue</h3>
      <div className="queue-table">
        {rows.map((row) => (
          <div className="queue-row" key={row.step}>
            <span>{row.step}</span>
            <strong>{row.box}</strong>
            <span>{row.z}</span>
            <em>{row.status}</em>
          </div>
        ))}
      </div>
    </section>
  );
}

function Calibration({ roi, pallet, setPallet, mount }) {
  function updatePallet(key, value) {
    setPallet((current) => ({
      ...current,
      [key]: value,
    }));
  }

  return (
    <section className="calibration-grid">
      <article className="cal-card">
        <p className="label">Mounting</p>
        <h3>Recommended height</h3>
        <strong>{mount.recommendedHeight} mm</strong>
        <p>Minimum {mount.minHeight} mm from pallet top for {mount.palletWidth} x {mount.palletDepth} mm, using conservative D457 depth FOV H {mount.conservativeH} deg / V {mount.conservativeV} deg.</p>
      </article>
      <article className="cal-card">
        <p className="label">Pallet footprint</p>
        <h3>Dimensions</h3>
        <div className="dimension-grid">
          <label>
            Width mm
            <input
              type="text"
              inputMode="decimal"
              value={pallet.width}
              onChange={(event) => updatePallet('width', event.target.value)}
            />
          </label>
          <label>
            Depth mm
            <input
              type="text"
              inputMode="decimal"
              value={pallet.depth}
              onChange={(event) => updatePallet('depth', event.target.value)}
            />
          </label>
        </div>
        <p>Width is across the camera horizontal FOV. Depth is across the camera vertical FOV.</p>
      </article>
      <article className="cal-card">
        <p className="label">Processing</p>
        <h3>Depth + zero-shot AI</h3>
        <strong>Non-teaching</strong>
        <p>Real mode aligns depth to color, masks outside the ROI, segments boxes with FastSAM or YOLOv8-seg, then uses depth for every candidate height and 3D pick point. A depth-edge fallback can be selected from the Python CLI.</p>
      </article>
      <article className="cal-card">
        <p className="label">ROI</p>
        <h3>Active drag zone</h3>
        <strong>{Math.round(roi.w)}% x {Math.round(roi.h)}%</strong>
        <p>Drag in the Vision page to redefine the pallet boundary.</p>
      </article>
      <article className="cal-card wide">
        <p className="label">Hand-eye matrix</p>
        <h3>Camera to robot transform</h3>
        <code>
          [ 1.000  0.000  0.000  0.0 ]<br />
          [ 0.000  1.000  0.000  0.0 ]<br />
          [ 0.000  0.000  1.000  0.0 ]<br />
          [ 0.000  0.000  0.000  1.0 ]
        </code>
        <p>Example: camera point [137, 27, 756, 1] multiplied by this 4x4 matrix returns robot base XYZ. Replace identity with your calibrated eye-to-hand transform before robot deployment.</p>
      </article>
    </section>
  );
}

function Metrics({ captured, detectedCount, mount, demoMode }) {
  return (
    <section className="metrics">
      <Metric label="Min mount Z" value={`${mount.minHeight} mm`} tone="lavender" />
      <Metric label="Recommended Z" value={`${mount.recommendedHeight} mm`} tone="lime" />
      <Metric label="Detected boxes" value={captured ? String(detectedCount) : '0'} tone="graphite" />
      <Metric label="Decision mode" value={demoMode ? 'Demo shot' : 'RealSense'} tone="red" />
    </section>
  );
}

function Metric({ label, value, tone }) {
  return (
    <article className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function Coord({ label, value }) {
  return (
    <div className="coord-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusPill({ label }) {
  return <span className="status-pill">{label}</span>;
}

createRoot(document.getElementById('root')).render(<App />);
