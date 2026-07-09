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
  { id: 1, x: 8, y: 9, w: 28, h: 21, z: 846, color: '#B99EF8' },
  { id: 2, x: 38, y: 9, w: 24, h: 21, z: 781, color: '#F2C96B' },
  { id: 3, x: 64, y: 9, w: 28, h: 21, z: 910, color: '#83A633' },
  { id: 4, x: 8, y: 34, w: 23, h: 25, z: 832, color: '#69BFE8' },
  { id: 5, x: 33, y: 34, w: 29, h: 25, z: 756, color: '#DFA1D8' },
  { id: 6, x: 64, y: 34, w: 28, h: 25, z: 836, color: '#B99EF8' },
  { id: 7, x: 8, y: 63, w: 31, h: 19, z: 901, color: '#DEF4A5' },
  { id: 8, x: 41, y: 63, w: 23, h: 19, z: 819, color: '#6AA9F2' },
  { id: 9, x: 66, y: 63, w: 26, h: 19, z: 874, color: '#F6A06E' },
];

const D457_DEPTH_FOV = { h: 87, v: 58, tolerance: 3 };
const D457_MIN_RANGE_MM = 600;
const MOUNT_MARGIN = 0.15;
const DEFAULT_PALLET = { width: '1100', depth: '1100' };

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

function estimateBoxDepth(areaRatio, centerX, centerY) {
  const centerPenalty = Math.hypot(centerX - 50, centerY - 50) * 1.8;
  const areaBonus = Math.min(areaRatio * 850, 280);
  return Math.round(940 - areaBonus + centerPenalty);
}

function buildMaskFromImage(imageData, width, height, roi) {
  const data = imageData.data;
  const mask = new Uint8Array(width * height);
  const startX = Math.max(0, Math.floor((roi.x / 100) * width));
  const startY = Math.max(0, Math.floor((roi.y / 100) * height));
  const endX = Math.min(width, Math.ceil(((roi.x + roi.w) / 100) * width));
  const endY = Math.min(height, Math.ceil(((roi.y + roi.h) / 100) * height));
  const sampleR = [];
  const sampleG = [];
  const sampleB = [];
  const borderStep = 4;
  const borderBand = Math.max(3, Math.round(Math.min(endX - startX, endY - startY) * 0.035));

  for (let y = startY; y < endY; y += borderStep) {
    for (let x = startX; x < endX; x += borderStep) {
      const nearBorder = x < startX + borderBand || x > endX - borderBand || y < startY + borderBand || y > endY - borderBand;
      if (!nearBorder) continue;
      const idx = (y * width + x) * 4;
      sampleR.push(data[idx]);
      sampleG.push(data[idx + 1]);
      sampleB.push(data[idx + 2]);
    }
  }

  const median = (items) => {
    if (!items.length) return 128;
    const sorted = [...items].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  const bgR = median(sampleR);
  const bgG = median(sampleG);
  const bgB = median(sampleB);
  let objectPixels = 0;

  for (let y = startY; y < endY; y += 1) {
    for (let x = startX; x < endX; x += 1) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const brightness = (r + g + b) / 3;
      const backgroundDistance = Math.hypot(r - bgR, g - bgG, b - bgB);
      const isCardboard =
        brightness > 36 &&
        brightness < 226 &&
        max - min > 24 &&
        r >= g * 0.94 &&
        g >= b * 0.86 &&
        r > b * 1.16;
      const isForegroundObject =
        backgroundDistance > 58 &&
        brightness > 22 &&
        brightness < 245 &&
        !(max - min < 10 && brightness > 205);
      if (isCardboard || isForegroundObject) {
        mask[y * width + x] = 1;
        objectPixels += 1;
      }
    }
  }

  const roiArea = Math.max(1, (endX - startX) * (endY - startY));
  if (objectPixels > roiArea * 0.004) {
    return { mask, bounds: { startX, startY, endX, endY }, roiArea };
  }

  for (let y = startY + 1; y < endY - 1; y += 1) {
    for (let x = startX + 1; x < endX - 1; x += 1) {
      const idx = (y * width + x) * 4;
      const left = idx - 4;
      const right = idx + 4;
      const up = idx - width * 4;
      const down = idx + width * 4;
      const grayX = Math.abs(data[right] + data[right + 1] + data[right + 2] - data[left] - data[left + 1] - data[left + 2]);
      const grayY = Math.abs(data[down] + data[down + 1] + data[down + 2] - data[up] - data[up + 1] - data[up + 2]);
      if (grayX + grayY > 96) {
        mask[y * width + x] = 1;
      }
    }
  }

  return { mask, bounds: { startX, startY, endX, endY }, roiArea };
}

function connectedComponents(mask, width, height, bounds, roiArea) {
  const visited = new Uint8Array(width * height);
  const boxesFound = [];
  const minArea = Math.max(160, roiArea * 0.012);
  const queue = [];

  for (let y = bounds.startY; y < bounds.endY; y += 1) {
    for (let x = bounds.startX; x < bounds.endX; x += 1) {
      const seed = y * width + x;
      if (!mask[seed] || visited[seed]) continue;

      let head = 0;
      let area = 0;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      queue.length = 0;
      queue.push(seed);
      visited[seed] = 1;

      while (head < queue.length) {
        const current = queue[head];
        head += 1;
        const cx = current % width;
        const cy = Math.floor(current / width);
        area += 1;
        minX = Math.min(minX, cx);
        maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy);
        maxY = Math.max(maxY, cy);

        for (let oy = -1; oy <= 1; oy += 1) {
          for (let ox = -1; ox <= 1; ox += 1) {
            if (ox === 0 && oy === 0) continue;
            const nx = cx + ox;
            const ny = cy + oy;
            if (nx < bounds.startX || nx >= bounds.endX || ny < bounds.startY || ny >= bounds.endY) continue;
            const next = ny * width + nx;
            if (mask[next] && !visited[next]) {
              visited[next] = 1;
              queue.push(next);
            }
          }
        }
      }

      const boxW = maxX - minX + 1;
      const boxH = maxY - minY + 1;
      const fillRatio = area / Math.max(1, boxW * boxH);
      const roiW = bounds.endX - bounds.startX;
      const roiH = bounds.endY - bounds.startY;
      const touchesRoiBorder =
        minX <= bounds.startX + 3 ||
        maxX >= bounds.endX - 4 ||
        minY <= bounds.startY + 3 ||
        maxY >= bounds.endY - 4;
      const looksLikeBorderNoise = touchesRoiBorder && (boxW > roiW * 0.45 || boxH > roiH * 0.45);
      const largeEnoughForCase = boxW > roiW * 0.07 && boxH > roiH * 0.1;
      if (area >= minArea && largeEnoughForCase && fillRatio > 0.08 && !looksLikeBorderNoise) {
        const xPct = (minX / width) * 100;
        const yPct = (minY / height) * 100;
        const wPct = (boxW / width) * 100;
        const hPct = (boxH / height) * 100;
        const centerX = xPct + wPct / 2;
        const centerY = yPct + hPct / 2;
        boxesFound.push({
          x: xPct,
          y: yPct,
          w: wPct,
          h: hPct,
          z: estimateBoxDepth(area / roiArea, centerX, centerY),
          color: '#def4a5',
          source: 'actual-rgb',
        });
      }
    }
  }

  return boxesFound
    .sort((a, b) => (b.w * b.h) - (a.w * a.h))
    .slice(0, 12)
    .sort((a, b) => (a.y - b.y) || (a.x - b.x))
    .map((item, index) => ({ ...item, id: index + 1 }));
}

function buildHighContrastMask(imageData, width, bounds) {
  const data = imageData.data;
  const mask = new Uint8Array(width * Math.ceil(data.length / (width * 4)));

  for (let y = bounds.startY; y < bounds.endY; y += 1) {
    for (let x = bounds.startX; x < bounds.endX; x += 1) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const brightness = (r + g + b) / 3;
      const darkObject = brightness < 105 && max < 150;
      const saturatedObject = brightness > 58 && brightness < 235 && max - min > 48 && max > 115;
      const cardboardYellow = r > 120 && g > 78 && b < 150 && r > b * 1.18 && g > b * 0.92;
      if (darkObject || saturatedObject || cardboardYellow) {
        mask[y * width + x] = 1;
      }
    }
  }

  return mask;
}

function analyzeCameraFrame(video, roi) {
  if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return { boxes: [], imageUrl: '', message: 'Camera frame not ready' };
  }

  const width = 640;
  const displayWidth = video.clientWidth || video.videoWidth;
  const displayHeight = video.clientHeight || video.videoHeight;
  const height = Math.max(360, Math.round(width * (displayHeight / displayWidth)));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  const coverScale = Math.max(width / video.videoWidth, height / video.videoHeight);
  const sourceWidth = width / coverScale;
  const sourceHeight = height / coverScale;
  const sourceX = (video.videoWidth - sourceWidth) / 2;
  const sourceY = (video.videoHeight - sourceHeight) / 2;
  ctx.drawImage(video, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const { mask, bounds, roiArea } = buildMaskFromImage(imageData, width, height, roi);
  let boxesFound = connectedComponents(mask, width, height, bounds, roiArea);
  if (!boxesFound.length) {
    boxesFound = connectedComponents(buildHighContrastMask(imageData, width, bounds), width, height, bounds, roiArea);
  }

  return {
    boxes: boxesFound,
    imageUrl: canvas.toDataURL('image/jpeg', 0.86),
    message: boxesFound.length ? `Actual RGB detected ${boxesFound.length} box(es)` : 'Actual RGB found no box in ROI',
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
  const videoElementRef = useRef(null);

  const mount = useMemo(() => calculateMountingHeight(pallet), [pallet]);
  const demoDetectedBoxes = useMemo(() => boxes.filter((item) => isBoxInsideRoi(item, roi)), [roi]);
  const detectedBoxes = demoMode ? demoDetectedBoxes : actualBoxes;
  const highestBox = useMemo(() => chooseHighest(detectedBoxes), [detectedBoxes]);
  const selectedBox = detectedBoxes.find((item) => item.id === selectedId) ?? null;
  const candidate = selectedBox ?? (captured ? highestBox : null);

  useEffect(() => {
    let active = true;
    const cameraTimeout = window.setTimeout(() => {
      if (active && !cameraStream) setCameraStatus('Waiting for camera permission');
    }, 3500);

    setCameraStatus('Requesting RGB preview');
    if (!navigator.mediaDevices?.getUserMedia) {
      window.clearTimeout(cameraTimeout);
      setCameraStatus('Browser camera API unavailable');
      return;
    }

    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      .then((stream) => {
        window.clearTimeout(cameraTimeout);
        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
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
  }, []);

  function resetPick() {
    setCaptured(false);
    setConfirmed(false);
    setSelectedId(null);
    setActualBoxes([]);
    setCapturedFrame('');
  }

  function capture() {
    setCaptured(true);
    setConfirmed(false);
    if (demoMode) {
      setActualBoxes([]);
      setCapturedFrame('');
      setSelectedId(chooseHighest(demoDetectedBoxes)?.id ?? null);
      return;
    }

    const result = analyzeCameraFrame(videoElementRef.current, roi);
    setActualBoxes(result.boxes);
    setCapturedFrame(result.imageUrl);
    setCameraStatus(result.message);
    setSelectedId(chooseHighest(result.boxes)?.id ?? null);
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
              {demoMode ? 'Demo result on' : 'Actual RGB mode'}
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
                modeLabel={demoMode ? 'Simulated depth' : 'Actual RGB estimate'}
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
        <Coord label="Camera XYZ" value={candidate ? `${Math.round(candidate.x + candidate.w / 2)}, ${Math.round(candidate.y + candidate.h / 2)}, ${candidate.z}` : '--'} />
        <Coord label="Robot XYZ" value={candidate ? `${Math.round(candidate.x + candidate.w / 2)}, ${Math.round(candidate.y + candidate.h / 2)}, ${candidate.z}` : '--'} />
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

function PalletPreview({ captured, selectedId, detectedBoxes, onSelect, demoMode, cameraStream, cameraStatus, videoElementRef, capturedFrame, roi, setRoi }) {
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
        {!captured && cameraStream && <video ref={videoRef} className="camera-feed" autoPlay muted playsInline />}
        {!captured && !cameraStream && <div className="camera-empty">{cameraStatus}. Allow camera access to aim before capture.</div>}
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

        {captured && detectedBoxes.map((item) => {
          const active = captured && item.id === selectedId;
          return (
            <button
              key={item.id}
              className={`box-tile ${demoMode ? 'demo' : 'actual'} ${active ? 'active' : ''}`}
              style={{
                left: `${item.x}%`,
                top: `${item.y}%`,
                width: `${item.w}%`,
                height: `${item.h}%`,
                '--box-color': item.color,
              }}
                onClick={() => onSelect(item.id)}
            >
              <span>Box {item.id} - {demoMode ? 'Z' : 'Z est'} {item.z}mm</span>
            </button>
          );
        })}
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
      <Metric label="Decision mode" value={demoMode ? 'Demo shot' : 'Actual RGB'} tone="red" />
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
