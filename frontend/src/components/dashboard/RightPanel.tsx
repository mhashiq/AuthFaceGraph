import React from 'react';
import type { DLAnalysisResult, ExpertSystemResult, BehaviorResult } from '../../types/analysis';
import { Brain, Cpu, Sparkles, Smile, ShieldAlert, RefreshCw, BarChart2 } from 'lucide-react';
import clsx from 'clsx';

interface RightPanelProps {
  dl: DLAnalysisResult | undefined;
  expert: ExpertSystemResult | undefined;
  behavior: BehaviorResult | undefined;
}

const EMOTION_EMOJIS: Record<string, string> = {
  neutral: '😐',
  happy: '😊',
  sad: '😢',
  surprise: '😲',
  fear: '😨',
  disgust: '🤢',
  anger: '😠',
  contempt: '😏',
  unknown: '🤖',
};

const EMOTION_COLORS: Record<string, string> = {
  neutral: 'text-slate-400 border-slate-500/30 bg-slate-500/10',
  happy: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
  sad: 'text-blue-400 border-blue-500/30 bg-blue-500/10',
  surprise: 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10',
  fear: 'text-purple-400 border-purple-500/30 bg-purple-500/10',
  disgust: 'text-olive-400 border-olive-500/30 bg-olive-500/10',
  anger: 'text-rose-400 border-rose-500/30 bg-rose-500/10',
  contempt: 'text-amber-400 border-amber-500/30 bg-amber-500/10',
};

export const RightPanel: React.FC<RightPanelProps> = ({ dl, expert, behavior }) => {
  const ensemble = dl?.emotion_ensemble;
  const emotion = ensemble?.final_emotion?.toLowerCase() ?? 'unknown';
  const confidence = ensemble?.confidence ?? 0;
  const prevPrediction = ensemble?.model_predictions?.[0]?.emotion ?? 'Neutral';

  // Format Helper
  const pct = (val: number) => `${Math.round(val * 100)}%`;

  return (
    <aside className="space-y-5 h-full overflow-y-auto pr-1">
      {/* ── CARD 1: EMOTION ANALYTICS ──────────────────────────────────────── */}
      <div className="bg-dark-900 border border-dark-600/60 rounded-2xl p-5 flex flex-col gap-4 shadow-2xl">
        <div className="flex items-center gap-2 border-b border-dark-700/60 pb-3">
          <Smile size={16} className="text-violet-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono">
            Emotion Classifier
          </h3>
        </div>

        <div className="flex items-center gap-4 bg-slate-950/40 border border-dark-600/40 rounded-xl p-4">
          <div className="text-4xl filter drop-shadow-[0_0_10px_rgba(255,255,255,0.1)]">
            {EMOTION_EMOJIS[emotion] ?? EMOTION_EMOJIS.unknown}
          </div>
          <div className="flex-1 min-w-0">
            <span className="text-[9px] text-dark-400 font-mono uppercase tracking-wider">
              Predicted State
            </span>
            <h2 className="text-lg font-bold text-white font-mono capitalize leading-tight">
              {ensemble?.final_emotion ?? 'Neutral'}
            </h2>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="text-[10px] text-violet-400 font-bold font-mono">
                {pct(confidence)} Confidence
              </span>
            </div>
          </div>

          {/* Simple Circular confidence ring */}
          <div className="relative w-12 h-12 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
              <path
                className="text-slate-800"
                strokeWidth="3"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                className="text-violet-500 transition-all duration-500"
                strokeWidth="3.5"
                strokeDasharray={`${Math.round(confidence * 100)}, 100`}
                strokeLinecap="round"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
            <span className="absolute text-[8px] font-mono font-bold text-white">
              {pct(confidence)}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[9px] font-mono bg-slate-950/20 border border-dark-600/30 p-2.5 rounded-lg">
          <div>
            <span className="text-dark-400 block">PREVIOUS STATE:</span>
            <span className="text-slate-200 capitalize font-semibold">{prevPrediction}</span>
          </div>
          <div>
            <span className="text-dark-400 block">DISSENSUS INDEX:</span>
            <span className="text-violet-400 font-bold">
              {ensemble?.disagreement_score !== undefined ? ensemble.disagreement_score.toFixed(3) : '0.000'}
            </span>
          </div>
        </div>
      </div>

      {/* ── CARD 2: EXPERT SYSTEM ─────────────────────────────────────────── */}
      <div className="bg-dark-900 border border-dark-600/60 rounded-2xl p-5 flex flex-col gap-4 shadow-2xl">
        <div className="flex items-center gap-2 border-b border-dark-700/60 pb-3">
          <Brain size={16} className="text-violet-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono">
            Physiological Expert System
          </h3>
        </div>

        {/* Attention indicator */}
        <div className="flex justify-between items-center bg-slate-950/40 border border-dark-600/40 p-3 rounded-xl">
          <span className="text-[10px] text-dark-300 font-mono uppercase font-bold">
            Attention State
          </span>
          <span className={clsx(
            'px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase border',
            expert?.attention_state === 'focused' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
            expert?.attention_state === 'distracted' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
            expert?.attention_state === 'drowsy' ? 'bg-rose-500/10 text-rose-400 border-rose-500/20 animate-pulse' :
            'bg-dark-700 text-dark-300'
          )}>
            {expert?.attention_state ?? 'focused'}
          </span>
        </div>

        {/* Composite Progress Indicators */}
        <div className="space-y-3 font-mono text-[10px]">
          {/* Focus */}
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">FOCUS INDEX</span>
              <span className="text-emerald-400 font-bold">{pct(expert?.focus_score ?? 1.0)}</span>
            </div>
            <div className="h-1.5 w-full bg-slate-950/50 rounded-full overflow-hidden border border-dark-600/20">
              <div 
                className="h-full bg-gradient-to-r from-emerald-600 to-teal-400 rounded-full transition-all duration-300"
                style={{ width: `${Math.round((expert?.focus_score ?? 1.0) * 100)}%` }}
              />
            </div>
          </div>

          {/* Fatigue */}
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">FATIGUE INDEX</span>
              <span className="text-rose-400 font-bold">{pct(expert?.fatigue_score ?? 0.0)}</span>
            </div>
            <div className="h-1.5 w-full bg-slate-950/50 rounded-full overflow-hidden border border-dark-600/20">
              <div 
                className="h-full bg-gradient-to-r from-rose-600 to-red-400 rounded-full transition-all duration-300"
                style={{ width: `${Math.round((expert?.fatigue_score ?? 0.0) * 100)}%` }}
              />
            </div>
          </div>

          {/* Landmark Stability */}
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">LANDMARK STABILITY</span>
              <span className="text-purple-400 font-bold">{pct(behavior?.landmark_stability ?? 1.0)}</span>
            </div>
            <div className="h-1.5 w-full bg-slate-950/50 rounded-full overflow-hidden border border-dark-600/20">
              <div 
                className="h-full bg-gradient-to-r from-purple-600 to-indigo-400 rounded-full transition-all duration-300"
                style={{ width: `${Math.round((behavior?.landmark_stability ?? 1.0) * 100)}%` }}
              />
            </div>
          </div>
        </div>

        {/* Risk Alerts list */}
        {expert?.alerts && expert.alerts.length > 0 && (
          <div className="border-t border-dark-700/30 pt-3">
            <span className="text-[9px] text-dark-400 font-mono uppercase tracking-wider block mb-2">
              Risk Level Alerts
            </span>
            <div className="space-y-1.5">
              {expert.alerts.map((alert: string, i: number) => {
                const isCritical = alert.toLowerCase().includes('microsleep') || alert.toLowerCase().includes('critical') || alert.toLowerCase().includes('closed');
                return (
                  <div 
                    key={i} 
                    className={clsx(
                      'p-2 border rounded-lg text-[10px] font-mono',
                      isCritical ? 'bg-rose-500/10 border-rose-500/20 text-rose-400 animate-pulse' :
                      'bg-amber-500/10 border-amber-500/20 text-amber-400'
                    )}
                  >
                    {alert}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ── CARD 3: EXPLAINABLE AI (XAI) ──────────────────────────────────── */}
      <div className="bg-dark-900 border border-dark-600/60 rounded-2xl p-5 flex flex-col gap-4 shadow-2xl">
        <div className="flex items-center gap-2 border-b border-dark-700/60 pb-3">
          <Sparkles size={16} className="text-violet-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono">
            Decision Explainability (XAI)
          </h3>
        </div>

        {dl?.xai_explanations && dl.xai_explanations.length > 0 ? (
          dl.xai_explanations.map((exp, i) => (
            <div key={i} className="flex flex-col gap-3 font-mono text-[10px]">
              {/* Text explanation */}
              <div className="bg-slate-950/40 border border-dark-600/30 rounded-xl p-3.5 text-slate-300 leading-relaxed shadow-inner">
                {exp.explanation_text || "No explanatory attributions."}
              </div>

              {/* Attribution progress bars */}
              {exp.attributions && exp.attributions.length > 0 && (
                <div className="space-y-2">
                  <span className="text-dark-400 block font-semibold mb-1">
                    FEATURE GROUP CONTRIBS:
                  </span>
                  {exp.attributions.slice(0, 3).map((attr, attrIdx) => {
                    const cleanName = attr.feature_name.replace('region_', '').toUpperCase();
                    const percentage = Math.round(attr.contribution * 100);
                    return (
                      <div key={attrIdx} className="space-y-1">
                        <div className="flex justify-between text-[9px]">
                          <span className="text-slate-300">{cleanName}</span>
                          <span className="text-violet-400 font-bold">{percentage}%</span>
                        </div>
                        <div className="h-1 w-full bg-slate-950/50 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-gradient-to-r from-violet-600 to-indigo-400 rounded-full"
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))
        ) : (
          <p className="text-dark-400 font-mono text-[10px] text-center py-4">
            Awaiting GNN graph explanations...
          </p>
        )}
      </div>
    </aside>
  );
};
