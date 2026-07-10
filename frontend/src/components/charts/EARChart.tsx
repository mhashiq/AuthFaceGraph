/**
 * AuthBrain AI Face Analysis Engine
 * Real-time chart components:
 *   - EARChart      — Eye Aspect Ratio (left / right / avg)
 *   - HeadPoseChart — Pitch / Yaw / Roll over time
 *   - FatigueChart  — Fatigue & Focus score over time
 */

import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
} from 'recharts';
import { useAnalysisStore } from '../../store';

// ── Shared dark tooltip ────────────────────────────────────────────────────────
const DarkTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-dark-800 border border-dark-600 rounded-lg p-2 text-xs font-mono shadow-xl">
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
        </div>
      ))}
    </div>
  );
};

const TICK_STYLE = { fill: '#6b7280', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' };


// ══════════════════════════════════════════════════════════════════════════════
// EAR Chart — Eye Aspect Ratio
// ══════════════════════════════════════════════════════════════════════════════

export const EARChart: React.FC = () => {
  const earHistory = useAnalysisStore((s) => s.earHistory);
  const data = earHistory.slice(-60);

  return (
    <div className="bg-dark-800/40 border border-dark-600/50 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold font-mono text-white uppercase tracking-wider">
          Eye Aspect Ratio (EAR)
        </h3>
        <span className="text-xs font-mono text-dark-400">{data.length} frames</span>
      </div>

      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis dataKey="time" hide />
          <YAxis domain={[0, 0.5]} tick={TICK_STYLE} tickCount={4} />
          <Tooltip content={<DarkTooltip />} />
          <ReferenceLine
            y={0.25}
            stroke="#ff2d55"
            strokeDasharray="4 2"
            strokeWidth={1}
            label={{ value: 'Blink ←', fill: '#ff2d55', fontSize: 9 }}
          />
          <Line type="monotone" dataKey="left_ear"  stroke="#00d4ff" strokeWidth={1.5} dot={false} name="Left EAR"  isAnimationActive={false} />
          <Line type="monotone" dataKey="right_ear" stroke="#bf5af2" strokeWidth={1.5} dot={false} name="Right EAR" isAnimationActive={false} />
          <Line type="monotone" dataKey="avg_ear"   stroke="#00ff41" strokeWidth={2}   dot={false} name="Avg EAR"   isAnimationActive={false} />
          <Legend iconSize={8} iconType="line" wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace', paddingTop: 8 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};


// ══════════════════════════════════════════════════════════════════════════════
// Head Pose Chart — Pitch / Yaw / Roll
// ══════════════════════════════════════════════════════════════════════════════

export const HeadPoseChart: React.FC = () => {
  const history = useAnalysisStore((s) => s.headPoseHistory);
  const data = history.slice(-60);

  return (
    <div className="bg-dark-800/40 border border-dark-600/50 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold font-mono text-white uppercase tracking-wider">
          Head Pose Angles
        </h3>
        <div className="flex gap-3 text-xs font-mono">
          <span className="text-[#00d4ff]">Pitch</span>
          <span className="text-[#bf5af2]">Yaw</span>
          <span className="text-[#ff9f0a]">Roll</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={130}>
        <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
          <defs>
            <linearGradient id="pitchGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#00d4ff" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="yawGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#bf5af2" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#bf5af2" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis dataKey="time" hide />
          <YAxis domain={[-45, 45]} tick={TICK_STYLE} tickCount={5} />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
          <Tooltip content={<DarkTooltip />} />
          <Area type="monotone" dataKey="pitch" stroke="#00d4ff" fill="url(#pitchGrad)" strokeWidth={1.5} dot={false} name="Pitch" isAnimationActive={false} />
          <Area type="monotone" dataKey="yaw"   stroke="#bf5af2" fill="url(#yawGrad)"   strokeWidth={1.5} dot={false} name="Yaw"   isAnimationActive={false} />
          <Line  type="monotone" dataKey="roll"  stroke="#ff9f0a" strokeWidth={1.5} dot={false} name="Roll" isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};


// ══════════════════════════════════════════════════════════════════════════════
// Fatigue / Focus Chart
// ══════════════════════════════════════════════════════════════════════════════

export const FatigueChart: React.FC = () => {
  const history = useAnalysisStore((s) => s.fatigueHistory);
  const data = history.slice(-60);

  return (
    <div className="bg-dark-800/40 border border-dark-600/50 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold font-mono text-white uppercase tracking-wider">
          Fatigue &amp; Focus Score
        </h3>
        <div className="flex gap-3 text-xs font-mono">
          <span className="text-[#ff453a]">Fatigue</span>
          <span className="text-[#64d2ff]">Focus</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={130}>
        <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
          <defs>
            <linearGradient id="fatigueGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#ff453a" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#ff453a" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="focusGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#64d2ff" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#64d2ff" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis dataKey="time" hide />
          <YAxis domain={[0, 1]} tick={TICK_STYLE} tickCount={3} />
          <Tooltip content={<DarkTooltip />} />
          <Area type="monotone" dataKey="fatigue" stroke="#ff453a" fill="url(#fatigueGrad)" strokeWidth={2} dot={false} name="Fatigue" isAnimationActive={false} />
          <Area type="monotone" dataKey="focus"   stroke="#64d2ff" fill="url(#focusGrad)"   strokeWidth={2} dot={false} name="Focus"   isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
