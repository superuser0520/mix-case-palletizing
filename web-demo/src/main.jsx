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

function App() {
  const [view, setView] = useState('vision');
  const [demoMode, setDemoMode] = useState(true);
  const [cameraStatus, setCameraStatus] = useState('Demo mode');
  const [captured, setCaptured] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [roi, setRoi] = useState({ x: 6, y: 6, w: 88, h: 88 });
  const [cameraStream, setCameraStream] = useState(null);

  const highestBox = useMemo(() => boxes.reduce((best, item) => (item.z < best.z ? item : best), boxes[0]), []);
  const selectedBox = boxes.find((item) => item.id === selectedId) ?? null;
  const candidate = selectedBox ?? (captured ? highestBox : null);

  useEffect(() => {
    if (demoMode) {
      cameraStream?.getTracks().forEach((track) => track.stop());
      setCameraStream(null);
      setCameraStatus('Demo mode');
      return;
    }

    let active = true;
    const cameraTimeout = window.setTimeout(() => {
      if (active && !cameraStream) setCameraStatus('Waiting for camera permission');
    }, 3500);

    setCameraStatus('Requesting camera');
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
        setCameraStatus('Browser camera live');
      })
      .catch(() => {
        window.clearTimeout(cameraTimeout);
        if (active) setCameraStatus('No browser camera permission');
      });

    return () => {
      active = false;
      window.clearTimeout(cameraTimeout);
    };
  }, [demoMode]);

  function resetPick() {
    setCaptured(false);
    setConfirmed(false);
    setSelectedId(null);
  }

  function capture() {
    setCaptured(true);
    setConfirmed(false);
    setSelectedId(highestBox.id);
  }

  function nextCandidate() {
    if (!captured) return;
    const currentIndex = boxes.findIndex((item) => item.id === (candidate?.id ?? highestBox.id));
    setSelectedId(boxes[(currentIndex + 1) % boxes.length].id);
    setConfirmed(false);
  }

  return (
    <main className="app-shell">
      <aside className="side-rail">
        <div className="brand">
          <div className="brand-mark"><ScanLine size={22} /></div>
          <div>
            <h1>PalletSight D457</h1>
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
            <StatusPill label={demoMode ? 'Demo mode' : cameraStatus} />
            <button className="toggle" onClick={() => { setDemoMode(!demoMode); resetPick(); }}>
              <ToggleLeft size={20} />
              {demoMode ? 'Demo on' : 'Demo off'}
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
                  demoMode={demoMode}
                  cameraStream={cameraStream}
                  cameraStatus={cameraStatus}
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
                onCapture={capture}
                onNext={nextCandidate}
                onConfirm={() => captured && setConfirmed(true)}
                onRedrag={resetPick}
              />
            </div>

            <Metrics captured={captured} />
          </>
        )}

        {view === 'queue' && <PickQueue candidate={candidate} confirmed={confirmed} />}
        {view === 'calibration' && <Calibration roi={roi} />}
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

function CandidatePanel({ captured, confirmed, candidate, onCapture, onNext, onConfirm, onRedrag }) {
  return (
    <aside className="candidate-panel">
      <div className="candidate-top">
        <p className="label">Pick Candidate</p>
        <h3>{candidate ? `Box ${candidate.id}` : 'No capture yet'}</h3>
        <p>{candidate ? 'Highest box proposed. Operator may override.' : 'Click Capture to evaluate the pallet.'}</p>
      </div>

      <div className="ring-card">
        <div className="ring"><Crosshair size={32} /></div>
        <div>
          <span>Highest box</span>
          <strong>{candidate ? `${candidate.z} mm` : '--'}</strong>
        </div>
      </div>

      <div className="coords">
        <Coord label="Camera XYZ" value={candidate ? `137, 27, ${candidate.z}` : '--'} />
        <Coord label="Robot XYZ" value={candidate ? `137, 27, ${candidate.z}` : '--'} />
      </div>

      <div className="actions">
        <button className="primary" onClick={onCapture}><MousePointer2 size={18} />Capture</button>
        <button className="secondary" disabled={!captured} onClick={onNext}><ChevronRight size={18} />Next</button>
        <button className="confirm" disabled={!captured} onClick={onConfirm}><Check size={18} />Confirm</button>
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

function PalletPreview({ captured, selectedId, onSelect, demoMode, cameraStream, cameraStatus, roi, setRoi }) {
  const stageRef = useRef(null);
  const videoRef = useRef(null);
  const dragStart = useRef(null);
  const [draftRoi, setDraftRoi] = useState(null);

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
    }
  }, [cameraStream]);

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

  function finishDrag() {
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
        {!demoMode && cameraStream && <video ref={videoRef} className="camera-feed" autoPlay muted playsInline />}
        {!demoMode && !cameraStream && <div className="camera-empty">{cameraStatus}. Use Demo On for synthetic pallet preview.</div>}

        <div
          className="roi-box"
          style={{
            left: `${shownRoi.x}%`,
            top: `${shownRoi.y}%`,
            width: `${shownRoi.w}%`,
            height: `${shownRoi.h}%`,
          }}
        />

        {demoMode && boxes.map((item) => {
          const active = captured && item.id === selectedId;
          return (
            <button
              key={item.id}
              className={`box-tile ${active ? 'active' : ''}`}
              style={{
                left: `${item.x}%`,
                top: `${item.y}%`,
                width: `${item.w}%`,
                height: `${item.h}%`,
                '--box-color': item.color,
              }}
              onClick={() => onSelect(item.id)}
            >
              {captured && <span>Box {item.id} - Z {item.z}mm</span>}
            </button>
          );
        })}
      </div>
      {!captured && <div className="capture-hint">Drag ROI directly on the preview. Processing starts only when Capture is pressed.</div>}
    </div>
  );
}

function PickQueue({ candidate, confirmed }) {
  const rows = [
    { step: 'Current candidate', box: candidate ? `Box ${candidate.id}` : 'Waiting', z: candidate ? `${candidate.z} mm` : '--', status: confirmed ? 'Confirmed' : 'Review' },
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

function Calibration({ roi }) {
  return (
    <section className="calibration-grid">
      <article className="cal-card">
        <p className="label">Mounting</p>
        <h3>Recommended height</h3>
        <strong>1215 mm</strong>
        <p>Computed from conservative D457 depth FOV for an 1100 mm pallet.</p>
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
      </article>
    </section>
  );
}

function Metrics({ captured }) {
  return (
    <section className="metrics">
      <Metric label="Min mount Z" value="1057 mm" tone="lavender" />
      <Metric label="Recommended Z" value="1215 mm" tone="lime" />
      <Metric label="Detected boxes" value={captured ? '9' : '0'} tone="graphite" />
      <Metric label="Decision mode" value="One-shot" tone="red" />
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
