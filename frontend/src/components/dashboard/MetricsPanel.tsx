/**
 * AuthBrain AI Face Analysis Engine
 * MetricsPanel Component
 *
 * Displays all real-time numerical measurements from the analysis pipeline
 * in a grid of animated metric cards.
 */

import React from 'react';
import { Eye, Activity, Smile, Brain, Compass, Zap, Shield, Clock } from 'lucide-react';
import { useAnalysisStore } from '../../store';
import clsx from 'clsx';
import type { Landmark, DLAnalysisResult } from '../../types/analysis';

// ── Metric Card ────────────────────────────────────────────────────────────────

interface MetricCardProps {
  label:    string;
  value:    string | number;
  unit?:    string;
  icon:     React.ReactNode;
  color?:   string;
  subValue?: string;
  pulse?:   boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  label, value, unit, icon, color = 'text-brand-500', subValue, pulse,
}) => (
  <div className={clsx(
    'bg-dark-800/60 border border-dark-600/50 rounded-xl p-4 flex flex-col gap-2',
    'hover:border-dark-500/80 transition-all duration-300',
    'backdrop-blur-sm',
  )}>
    <div className="flex items-center justify-between">
      <span className="text-dark-300 text-xs font-mono uppercase tracking-wider">{label}</span>
      <span className={clsx('opacity-70', color, pulse && 'animate-pulse-slow')}>{icon}</span>
    </div>
    <div className="flex items-baseline gap-1">
      <span className={clsx('text-2xl font-bold font-mono', color)}>{value}</span>
      {unit && <span className="text-dark-400 text-xs font-mono">{unit}</span>}
    </div>
    {subValue && (
      <span className="text-dark-400 text-xs font-mono truncate">{subValue}</span>
    )}
  </div>
);

// ── Score Bar ──────────────────────────────────────────────────────────────────

interface ScoreBarProps {
  label: string;
  value: number;  // 0–1
  color: string;
  inverse?: boolean;  // If true, high = bad (fatigue)
}

const ScoreBar: React.FC<ScoreBarProps> = ({ label, value, color, inverse }) => {
  const pct    = Math.round(value * 100);
  const isHigh = inverse ? value > 0.7 : value > 0.7;
  const warningColor = inverse
    ? value > 0.7 ? 'bg-risk-critical' : value > 0.4 ? 'bg-risk-medium' : 'bg-risk-low'
    : value > 0.7 ? 'bg-metric-focus' : value > 0.4 ? 'bg-yellow-500' : 'bg-risk-critical';

  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <span className="text-dark-300 text-xs font-mono">{label}</span>
        <span className={clsx('text-xs font-mono font-bold', color)}>{pct}%</span>
      </div>
      <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
        <div
          className={clsx('h-full rounded-full transition-all duration-300', warningColor)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
};

// ── Attention Badge ────────────────────────────────────────────────────────────

const ATTENTION_CONFIG: Record<string, { label: string; cls: string }> = {
  focused:    { label: '🎯 FOCUSED',    cls: 'bg-brand-500/20 text-brand-500 border-brand-500/40' },
  alert:      { label: '⚡ ALERT',      cls: 'bg-metric-focus/20 text-metric-focus border-metric-focus/40' },
  distracted: { label: '↩️ DISTRACTED', cls: 'bg-risk-medium/20 text-risk-medium border-risk-medium/40' },
  drowsy:     { label: '😴 DROWSY',     cls: 'bg-risk-critical/20 text-risk-critical border-risk-critical/40 animate-pulse' },
  unknown:    { label: '❓ UNKNOWN',    cls: 'bg-dark-600/40 text-dark-300 border-dark-500/40' },
};

const AttentionBadge: React.FC<{ state: string }> = ({ state }) => {
  const cfg = ATTENTION_CONFIG[state] ?? ATTENTION_CONFIG.unknown;
  return (
    <span className={clsx('inline-flex items-center px-3 py-1 rounded-full text-sm font-mono font-semibold border', cfg.cls)}>
      {cfg.label}
    </span>
  );
};


// ── GNN Graph Node Visualizer ──────────────────────────────────────────────────

interface GNNVisualizerProps {
  dl: DLAnalysisResult;
}

const GNNVisualizer: React.FC<GNNVisualizerProps> = ({ dl }) => {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const ensemble = dl.emotion_ensemble;

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!dl.landmarks || dl.landmarks.length === 0) {
      // Draw scanning indicator lines
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 20; i < canvas.width; i += 20) {
        ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height);
        ctx.moveTo(0, i); ctx.lineTo(canvas.width, i);
      }
      ctx.stroke();
      return;
    }

    const lms = dl.landmarks;

    // Find bounds of face to center and scale
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    lms.forEach((lm: Landmark) => {
      if (lm.x < minX) minX = lm.x;
      if (lm.x > maxX) maxX = lm.x;
      if (lm.y < minY) minY = lm.y;
      if (lm.y > maxY) maxY = lm.y;
    });

    const faceW = maxX - minX;
    const faceH = maxY - minY;
    
    // Zoom factor: small padding (3%) crops closely to the face extremes, magnifying it on the canvas
    const padding = 0.03;

    const scaleX = canvas.width / (faceW + padding * 2);
    const scaleY = canvas.height / (faceH + padding * 2);
    const scale = Math.min(scaleX, scaleY);

    const offsetX = (canvas.width - faceW * scale) / 2 - minX * scale;
    const offsetY = (canvas.height - faceH * scale) / 2 - minY * scale;

    // GNN Connection Lines (Opacity & thickness modulated by endpoint GNN node importance)
    const connections = [
      // Face outline / oval
      [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356],
      [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379],
      [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149],
      [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234],
      [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
      // Left Eye
      [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133],
      [133, 173], [173, 157], [157, 158], [158, 159], [159, 160], [160, 161], [161, 246], [246, 33],
      // Right Eye
      [362, 382], [382, 381], [381, 380], [380, 374], [374, 373], [373, 390], [390, 249],
      [249, 263], [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384],
      [384, 398], [398, 362],
      // Left Eyebrow
      [70, 63], [63, 105], [105, 66], [66, 107], [107, 55], [55, 117], [117, 124], [124, 70],
      // Right Eyebrow
      [300, 293], [293, 334], [334, 296], [296, 336], [336, 285], [285, 346], [346, 353], [353, 300],
      // Nose Bridge & Base
      [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19], [19, 94], [94, 2],
      [2, 323], [323, 356], [2, 93], [93, 132], [94, 324], [324, 323], [94, 141], [141, 93],
      // Lips / Mouth
      [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270],
      [270, 409], [409, 291], [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314],
      [314, 405], [405, 321], [321, 375], [375, 291]
    ];

    // GNN Connection Lines (Clean, uniform indigo mesh)
    connections.forEach(([p1, p2]) => {
      if (lms[p1] && lms[p2]) {
        const x1 = lms[p1].x * scale + offsetX;
        const y1 = lms[p1].y * scale + offsetY;
        const x2 = lms[p2].x * scale + offsetX;
        const y2 = lms[p2].y * scale + offsetY;

        ctx.strokeStyle = 'rgba(99, 102, 241, 0.32)'; // Clean sleek indigo
        ctx.lineWidth = 0.5;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    });

    // GNN Nodes: Clean, uniform indigo/violet nodes (no rose importance overlays)
    lms.forEach((lm: Landmark) => {
      const cx = lm.x * scale + offsetX;
      const cy = lm.y * scale + offsetY;

      ctx.fillStyle = 'rgba(99, 102, 241, 0.65)'; // Crisp indigo nodes
      ctx.beginPath();
      ctx.arc(cx, cy, 1.1, 0, 2 * Math.PI);
      ctx.fill();
    });
  }, [dl?.landmarks]);

  if (!ensemble) return null;

  return (
    <div className="bg-dark-800/40 border border-dark-600/50 rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-dark-300 text-xs font-mono uppercase tracking-wider">
          Predicted Emotion
        </span>
        <span className="text-sm font-bold text-violet-400 font-mono capitalize">
          {ensemble.final_emotion}
        </span>
      </div>
      
      <ScoreBar
        label="Emotion Confidence"
        value={ensemble.confidence}
        color="text-violet-400"
      />

      {/* GNN Graph visualizer */}
      <div className="mt-2 bg-slate-950/50 border border-slate-800/60 rounded-xl p-3 flex flex-col items-center gap-2">
        <span className="text-[10px] font-mono text-slate-500 uppercase self-start">
          GNN Node Network (478 Landmarks)
        </span>
        <canvas
          ref={canvasRef}
          width={200}
          height={200}
          className="bg-slate-950/30 rounded-lg max-w-full"
        />
        <span className="text-[9px] text-slate-500 font-mono text-center">
          Topological Landmark Map
        </span>
      </div>

      {/* Real-time Emotion Probabilities */}
      <div className="mt-3 border-t border-dark-600/40 pt-3">
        <span className="text-[10px] text-dark-300 font-mono uppercase tracking-wider block mb-2.5">
          Emotion Probabilities
        </span>
        <div className="space-y-2">
          {Object.entries(ensemble.probabilities)
            .sort((a, b) => b[1] - a[1]) // highest first
            .map(([emotion, prob]) => {
              const percentage = Math.round(prob * 100);
              
              // Emotion-specific colors
              let barColor = "from-violet-600 to-indigo-400";
              let textColor = "text-violet-400";
              if (emotion === 'happy') { barColor = "from-emerald-600 to-teal-400"; textColor = "text-emerald-400"; }
              else if (emotion === 'sad') { barColor = "from-blue-600 to-cyan-400"; textColor = "text-blue-400"; }
              else if (emotion === 'anger') { barColor = "from-rose-600 to-red-400"; textColor = "text-rose-400"; }
              else if (emotion === 'neutral') { barColor = "from-slate-600 to-slate-400"; textColor = "text-slate-400"; }

              return (
                <div key={emotion} className="space-y-1">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className={`capitalize ${textColor} font-medium`}>{emotion}</span>
                    <span className={`${textColor} font-bold`}>{percentage}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-950/50 rounded-full overflow-hidden border border-dark-600/10">
                    <div 
                      className={`h-full bg-gradient-to-r ${barColor} rounded-full transition-all duration-300`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
};


// ── Main MetricsPanel ──────────────────────────────────────────────────────────

export const MetricsPanel: React.FC = () => {
  const latestResult = useAnalysisStore(s => s.latestResult);

  // Throttle display updates to 300ms to keep numbers readable without blinking
  const [displayResult, setDisplayResult] = React.useState<typeof latestResult>(null);
  const lastUpdateRef = React.useRef<number>(0);

  React.useEffect(() => {
    if (!latestResult) return;
    const now = Date.now();
    if (now - lastUpdateRef.current > 300 || !displayResult) {
      setDisplayResult(latestResult);
      lastUpdateRef.current = now;
    }
  }, [latestResult]);

  const result = displayResult || latestResult;
  const r      = result;
  const eyes   = r?.eyes;
  const mouth  = r?.mouth;
  const pose   = r?.head_pose;
  const expert = r?.expert_system;
  const qual   = r?.quality;
  const behav  = r?.behavior;

  const na = (v: number | undefined, decimals = 2) =>
    v !== undefined ? v.toFixed(decimals) : '—';

  return (
    <div className="space-y-5">

      {/* Attention State */}
      <div className="bg-dark-800/40 border border-dark-600/50 rounded-xl p-4">
        <div className="flex items-center justify-between">
          <span className="text-dark-300 text-xs font-mono uppercase tracking-wider mb-2 block">
            Attention State
          </span>
          {expert && (
            <span className="text-xs font-mono text-dark-400">
              {Math.round((expert.overall_confidence ?? 0) * 100)}% conf.
            </span>
          )}
        </div>
        <AttentionBadge state={expert?.attention_state ?? behav?.attention_state ?? 'unknown'} />

        {expert && (
          <div className="mt-3 space-y-2">
            <ScoreBar label="Focus Score"   value={expert.focus_score}   color="text-metric-focus" />
            <ScoreBar label="Fatigue Score" value={expert.fatigue_score} color="text-metric-fatigue" inverse />
          </div>
        )}
      </div>

      {/* Deep Learning Emotion & GNN Graph Node Visualizer */}
      {r?.deep_learning?.dl_enabled && r.deep_learning.emotion_ensemble && (
        <GNNVisualizer dl={r.deep_learning} />
      )}

      {/* Core Metrics Grid */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="EAR (Avg)"
          value={na(eyes?.average_ear, 3)}
          icon={<Eye size={16} />}
          color="text-metric-ear"
          subValue={eyes?.gaze_direction ? `Gaze: ${eyes.gaze_direction}` : undefined}
        />
        <MetricCard
          label="Blink Count"
          value={eyes?.blink_count ?? '—'}
          icon={<Eye size={16} />}
          color="text-metric-blink"
          subValue={eyes ? `${eyes.blinks_per_minute.toFixed(1)} bpm` : undefined}
          pulse={eyes?.blink_detected}
        />
        <MetricCard
          label="Head Yaw"
          value={na(pose?.yaw, 1)}
          unit="°"
          icon={<Compass size={16} />}
          color="text-metric-pose"
          subValue={pose ? `P:${pose.pitch.toFixed(1)}° R:${pose.roll.toFixed(1)}°` : undefined}
        />
        <MetricCard
          label="Smile"
          value={mouth ? `${Math.round(mouth.smile_intensity * 100)}` : '—'}
          unit="%"
          icon={<Smile size={16} />}
          color="text-metric-smile"
          subValue={mouth?.yawn_detected ? '🥱 Yawn detected' : undefined}
        />
        <MetricCard
          label="Face Quality"
          value={qual ? `${Math.round(qual.overall_score * 100)}` : '—'}
          unit="%"
          icon={<Shield size={16} />}
          color={qual && qual.overall_score > 0.7 ? 'text-brand-500' : 'text-risk-medium'}
        />
        <MetricCard
          label="Inference"
          value={na(r?.inference_time_ms, 1)}
          unit="ms"
          icon={<Clock size={16} />}
          color={r && r.inference_time_ms < 33 ? 'text-brand-500' : 'text-risk-medium'}
          subValue={r ? `${r.fps.toFixed(1)} FPS` : undefined}
        />
        <MetricCard
          label="Symmetry"
          value={na(behav?.facial_symmetry ? behav.facial_symmetry * 100 : undefined, 1)}
          unit="%"
          icon={<Activity size={16} />}
          color="text-metric-ear"
        />
        <MetricCard
          label="Confidence"
          value={na(r?.model_confidence ? r.model_confidence * 100 : undefined, 1)}
          unit="%"
          icon={<Brain size={16} />}
          color="text-metric-focus"
          subValue={r?.face_detected ? `${r.face_count} face(s)` : 'No face'}
        />
      </div>

      {/* Eye Closure */}
      {eyes && eyes.eye_closure_duration_ms > 0 && (
        <div className="bg-risk-critical/10 border border-risk-critical/30 rounded-xl p-3 flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-risk-critical animate-pulse flex-shrink-0" />
          <span className="text-risk-critical text-xs font-mono">
            Eye closure: {eyes.eye_closure_duration_ms.toFixed(0)}ms
          </span>
        </div>
      )}
    </div>
  );
};
