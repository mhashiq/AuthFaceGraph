import React from 'react';
import { useAnalysisStore } from '../../store';
import { Shield, Brain, Clock, Activity, Cpu, AlertTriangle, Key } from 'lucide-react';
import clsx from 'clsx';

export const TopBar: React.FC = () => {
  const latestResult = useAnalysisStore(s => s.latestResult);
  const sessionId = useAnalysisStore(s => s.sessionId);
  const activeAlerts = useAnalysisStore(s => s.activeAlerts);

  const dl = latestResult?.deep_learning;
  const expert = latestResult?.expert_system;
  
  const emotion = dl?.emotion_ensemble?.final_emotion ?? 'Neutral';
  const confidence = dl?.emotion_ensemble?.confidence ?? 0.0;
  const attention = expert?.attention_state ?? latestResult?.behavior?.attention_state ?? 'focused';
  
  const getRiskLevel = (alerts: string[] | undefined) => {
    if (!alerts || alerts.length === 0) return 'low';
    const hasCritical = alerts.some(a => a.toLowerCase().includes('microsleep') || a.toLowerCase().includes('critical') || a.toLowerCase().includes('closed'));
    return hasCritical ? 'critical' : 'high';
  };
  const risk = getRiskLevel(expert?.alerts);

  const riskColors: Record<string, string> = {
    low:      'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    medium:   'text-yellow-400 bg-yellow-500/10 border-yellow-500/20',
    high:     'text-orange-400 bg-orange-500/10 border-orange-500/20',
    critical: 'text-rose-400 bg-rose-500/10 border-rose-500/20 animate-pulse',
  };

  return (
    <div className="bg-dark-900 border border-dark-600/60 rounded-2xl p-4 flex flex-col md:flex-row items-center justify-between gap-4 shadow-2xl">
      {/* Session info */}
      <div className="flex items-center gap-3 w-full md:w-auto">
        <div className="w-8 h-8 rounded-lg bg-violet-600/10 border border-violet-500/30 flex items-center justify-center">
          <Key size={14} className="text-violet-400" />
        </div>
        <div className="font-mono text-[10px]">
          <span className="text-dark-400 block uppercase tracking-wider">Active telemetry stream</span>
          <span className="text-slate-200 font-bold tracking-tight truncate max-w-[150px] sm:max-w-none block">
            {sessionId ? `SESSION: ${sessionId}` : 'WAITING FOR SOURCE...'}
          </span>
        </div>
      </div>

      {/* Main Stats metrics bar */}
      <div className="flex flex-wrap items-center gap-4 sm:gap-6 font-mono text-[10px]">
        {/* Current Emotion */}
        <div className="flex items-center gap-2 bg-slate-950/40 border border-dark-600/30 px-3 py-1.5 rounded-xl">
          <Brain size={12} className="text-violet-400" />
          <div>
            <span className="text-dark-400 block uppercase text-[8px]">Emotion</span>
            <span className="text-white font-bold capitalize">
              {emotion} ({Math.round(confidence * 100)}%)
            </span>
          </div>
        </div>

        {/* Attention State */}
        <div className="flex items-center gap-2 bg-slate-950/40 border border-dark-600/30 px-3 py-1.5 rounded-xl">
          <Activity size={12} className="text-violet-400" />
          <div>
            <span className="text-dark-400 block uppercase text-[8px]">Attention</span>
            <span className="text-white font-bold capitalize">{attention}</span>
          </div>
        </div>

        {/* Risk Level */}
        <div className="flex items-center gap-2 bg-slate-950/40 border border-dark-600/30 px-3 py-1.5 rounded-xl">
          <AlertTriangle size={12} className="text-violet-400" />
          <div>
            <span className="text-dark-400 block uppercase text-[8px]">Threat Risk</span>
            <span className={clsx('px-1.5 py-0.5 rounded border text-[9px] font-bold uppercase', riskColors[risk] || riskColors.low)}>
              {risk}
            </span>
          </div>
        </div>

        {/* Speed / FPS / Latency details */}
        <div className="hidden lg:flex items-center gap-5 border-l border-dark-700/60 pl-5">
          <div className="flex items-center gap-1.5">
            <Cpu size={12} className="text-dark-500" />
            <span className="text-slate-300 font-bold">{latestResult ? `${latestResult.fps.toFixed(1)} FPS` : '—'}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Clock size={12} className="text-dark-500" />
            <span className="text-slate-300 font-bold">{latestResult ? `${latestResult.inference_time_ms.toFixed(1)} ms` : '—'}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
