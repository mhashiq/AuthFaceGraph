/**
 * AuthFaceGraph — Premium TopBar
 * Real-time telemetry status bar with neon metrics
 */

import React, { useEffect, useState } from 'react';
import { useAnalysisStore } from '../../store';
import { Shield, Brain, Activity, AlertTriangle, Key, Clock, Zap } from 'lucide-react';
import { MetricBadge, StatusDot } from '../ui';
import clsx from 'clsx';

type Section = 'dashboard' | 'analytics' | 'users' | 'activity' | 'settings';

interface TopBarProps {
  activeSection: Section;
}

const SECTION_LABELS: Record<Section, string> = {
  dashboard:  'Live Analysis Dashboard',
  analytics:  'Analytics & Insights',
  users:      'User Management',
  activity:   'Activity & Monitoring',
  settings:   'System Settings',
};

export const TopBar: React.FC<TopBarProps> = ({ activeSection }) => {
  const latestResult = useAnalysisStore(s => s.latestResult);
  const sessionId    = useAnalysisStore(s => s.sessionId);
  const wsState      = useAnalysisStore(s => s.wsState);
  const [sessionTime, setSessionTime] = useState(0);
  const [startTime] = useState(() => Date.now());

  useEffect(() => {
    const interval = setInterval(() => setSessionTime(Math.floor((Date.now() - startTime) / 1000)), 1000);
    return () => clearInterval(interval);
  }, [startTime]);

  const formatTime = (s: number) => `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;

  const dl     = latestResult?.deep_learning;
  const expert = latestResult?.expert_system;

  const emotion     = dl?.emotion_ensemble?.final_emotion ?? 'Neutral';
  const confidence  = dl?.emotion_ensemble?.confidence ?? 0.0;
  const attention   = expert?.attention_state ?? latestResult?.behavior?.attention_state ?? 'Unknown';
  const fps         = latestResult?.fps ?? 0;
  const latency     = latestResult?.inference_time_ms ?? 0;

  const getRisk = () => {
    const alerts = expert?.alerts || [];
    if (!alerts.length) return { level: 'LOW', color: 'green' as const };
    const hasCrit = alerts.some(a => /microsleep|critical|closed/i.test(a));
    return hasCrit
      ? { level: 'CRITICAL', color: 'red' as const }
      : { level: 'ELEVATED', color: 'amber' as const };
  };
  const risk = getRisk();

  return (
    <header
      className="w-full rounded-2xl px-5 py-3 flex items-center justify-between gap-4 gpu-accelerated"
      style={{
        background: 'rgba(7,13,26,0.8)',
        backdropFilter: 'blur(24px)',
        border: '1px solid rgba(79,70,229,0.18)',
        boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
      }}
    >
      {/* ── LEFT: Section breadcrumb + session ─────────────────── */}
      <div className="flex items-center gap-4 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Shield size={14} style={{ color: '#8b5cf6' }} />
          <div>
            <div className="font-display text-[10px] font-bold tracking-widest uppercase"
              style={{ color: '#8b5cf6' }}>
              AuthFaceGraph
            </div>
            <div className="font-mono text-[10px] text-slate-400 tracking-wide leading-none mt-0.5">
              {SECTION_LABELS[activeSection]}
            </div>
          </div>
        </div>

        <div className="h-6 w-px bg-indigo-500/20" />

        {/* Session ID */}
        <div className="flex items-center gap-1.5">
          <Key size={10} className="text-violet-400/60" />
          <span className="font-mono text-[9px] text-slate-500 uppercase tracking-wider">
            {sessionId ? `SID:${sessionId.slice(0,8)}` : 'NO SESSION'}
          </span>
        </div>

        {/* Session timer */}
        <div className="flex items-center gap-1.5">
          <Clock size={10} className="text-cyan-400/60" />
          <span className="font-mono text-[9px] text-cyan-400/70 tabular-nums">
            {formatTime(sessionTime)}
          </span>
        </div>
      </div>

      {/* ── CENTER: Live metrics ────────────────────────────────── */}
      <div className="flex items-center gap-2 flex-wrap">
        <MetricBadge
          label="Emotion"
          value={`${emotion} ${Math.round(confidence * 100)}%`}
          icon={<Brain size={11} />}
          color="violet"
          animated={!!latestResult}
        />
        <MetricBadge
          label="Attention"
          value={attention}
          icon={<Activity size={11} />}
          color="cyan"
        />
        <MetricBadge
          label="Risk Level"
          value={risk.level}
          icon={<AlertTriangle size={11} />}
          color={risk.color}
        />
        <MetricBadge
          label="FPS"
          value={fps.toFixed(0)}
          icon={<Zap size={11} />}
          color={fps > 20 ? 'green' : 'amber'}
        />
        {latency > 0 && (
          <MetricBadge
            label="Latency"
            value={`${latency.toFixed(0)}ms`}
            color="blue"
          />
        )}
      </div>

      {/* ── RIGHT: WS status ───────────────────────────────────── */}
      <div className="flex items-center gap-3 flex-shrink-0">
        <StatusDot
          status={wsState === 'connected' ? 'online' : wsState === 'connecting' ? 'warning' : 'offline'}
          label={wsState === 'connected' ? 'Live' : wsState === 'connecting' ? 'Linking' : 'Offline'}
        />
      </div>
    </header>
  );
};
