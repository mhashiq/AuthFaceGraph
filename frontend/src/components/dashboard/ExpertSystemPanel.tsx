/**
 * AuthBrain AI Face Analysis Engine
 * ExpertSystemPanel Component
 *
 * Displays XAI explanations, feature attributions, alert history,
 * and expert system decisions from the backend inference engine.
 */

import React, { useState } from 'react';
import { Brain, ChevronDown, ChevronUp, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { useAnalysisStore } from '../../store';
import type { ExplanationResult, FeatureAttribution } from '../../types/analysis';
import clsx from 'clsx';

// ── Attribution Bar ────────────────────────────────────────────────────────────

const AttributionBar: React.FC<{ attr: FeatureAttribution }> = ({ attr }) => (
  <div className="space-y-0.5">
    <div className="flex justify-between items-center">
      <span className="text-dark-300 text-xs font-mono">{attr.feature_name.replace(/_/g, ' ')}</span>
      <span className="text-xs font-mono text-brand-500">{Math.round(attr.contribution * 100)}%</span>
    </div>
    <div className="h-1 bg-dark-700 rounded-full">
      <div
        className="h-full bg-gradient-to-r from-brand-700 to-brand-500 rounded-full transition-all duration-500"
        style={{ width: `${attr.contribution * 100}%` }}
      />
    </div>
    <p className="text-dark-500 text-[10px] font-mono leading-tight">{attr.description}</p>
  </div>
);

// ── Explanation Card ────────────────────────────────────────────────────────────

const ExplanationCard: React.FC<{ explanation: ExplanationResult }> = ({ explanation }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-dark-800/60 border border-dark-600/40 rounded-xl overflow-hidden">
      <button
        className="w-full flex items-center justify-between p-3 hover:bg-dark-700/40 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-brand-500" />
          <span className="text-sm font-mono text-white">{explanation.metric_name}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-dark-400">
            {Math.round(explanation.confidence * 100)}% conf
          </span>
          <span className="text-xs font-mono text-brand-500">
            {explanation.final_value.toFixed(3)}
          </span>
          {expanded ? <ChevronUp size={14} className="text-dark-400" /> : <ChevronDown size={14} className="text-dark-400" />}
        </div>
      </button>

      {expanded && (
        <div className="px-3 pb-3 space-y-3 border-t border-dark-700/50">
          {/* Explanation text */}
          <p className="text-dark-300 text-xs font-mono leading-relaxed pt-2">
            {explanation.explanation_text}
          </p>

          {/* Performance */}
          <div className="flex gap-4 text-xs font-mono">
            <span className="text-dark-500">
              Time: <span className="text-dark-300">{explanation.processing_time_ms.toFixed(1)}ms</span>
            </span>
            <span className="text-dark-500">
              LM Quality: <span className="text-dark-300">{Math.round(explanation.landmark_quality * 100)}%</span>
            </span>
          </div>

          {/* Feature attributions */}
          {explanation.attributions.length > 0 && (
            <div className="space-y-2">
              <span className="text-xs font-mono text-dark-500 uppercase tracking-wider">
                Feature Attributions
              </span>
              {explanation.attributions.map((attr) => (
                <AttributionBar key={attr.feature_name} attr={attr} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── Alert Badge ────────────────────────────────────────────────────────────────

const AlertBadge: React.FC<{ alert: string }> = ({ alert }) => {
  const isHigh    = alert.includes('⚠️') || alert.includes('Critical');
  const isMedium  = alert.includes('👁️') || alert.includes('↩️') || alert.includes('↔️');

  return (
    <div className={clsx(
      'flex items-start gap-2 p-2.5 rounded-lg border text-xs font-mono',
      isHigh   ? 'bg-risk-critical/10 border-risk-critical/30 text-risk-critical' :
      isMedium ? 'bg-risk-medium/10 border-risk-medium/30 text-risk-medium' :
                 'bg-risk-low/10 border-risk-low/30 text-risk-low',
    )}>
      {isHigh ? (
        <AlertTriangle size={12} className="flex-shrink-0 mt-0.5" />
      ) : (
        <Info size={12} className="flex-shrink-0 mt-0.5" />
      )}
      <span className="leading-tight">{alert}</span>
    </div>
  );
};
// ── Main ExpertSystemPanel ─────────────────────────────────────────────────────

export const ExpertSystemPanel: React.FC = () => {
  const result    = useAnalysisStore(s => s.latestResult);
  const activeAlerts = useAnalysisStore(s => s.activeAlerts);
  const wsState   = useAnalysisStore(s => s.wsState);

  // Maintain a cache of the last valid expert system results to avoid flickering/blinking
  const lastExpertRef = React.useRef<any>(null);
  
  if (result?.expert_system) {
    lastExpertRef.current = result.expert_system;
  }

  const activeExpert = result?.expert_system || lastExpertRef.current;
  const isWebcamRunning = wsState === 'connected';

  // Split throttling: scores update fast (400ms), explanations update slow (2000ms) to avoid flickering
  const [throttledScores, setThrottledScores] = React.useState<any>(null);
  const [throttledExplanations, setThrottledExplanations] = React.useState<any>(null);
  const lastScoreUpdateRef = React.useRef<number>(0);
  const lastExplUpdateRef = React.useRef<number>(0);

  React.useEffect(() => {
    if (!activeExpert) return;
    const now = Date.now();
    // Scores update every 400ms
    if (now - lastScoreUpdateRef.current > 400 || !throttledScores) {
      setThrottledScores({
        overall_confidence: activeExpert.overall_confidence,
        fatigue_score: activeExpert.fatigue_score,
        focus_score: activeExpert.focus_score,
        attention_state: activeExpert.attention_state,
        alerts: activeExpert.alerts,
      });
      lastScoreUpdateRef.current = now;
    }
    // Explanations update every 2000ms (text-heavy, avoid flicker)
    if (now - lastExplUpdateRef.current > 2000 || !throttledExplanations) {
      setThrottledExplanations({
        explanations: activeExpert.explanations,
      });
      lastExplUpdateRef.current = now;
    }
  }, [activeExpert]);

  const expert = throttledScores ? { ...throttledScores, explanations: throttledExplanations?.explanations || [] } : activeExpert;

  // If there's absolutely no data and we're not running, show the idle screen
  if (!expert && !isWebcamRunning) {
    return (
      <div className="bg-dark-800/40 border border-dark-600/50 rounded-xl p-6 flex flex-col items-center justify-center gap-3 text-center">
        <Brain size={32} className="text-dark-500" />
        <p className="text-dark-400 text-sm font-mono">Expert system idle</p>
        <p className="text-dark-500 text-xs font-mono">Start analysis to see XAI explanations</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Brain size={16} className="text-metric-focus" />
        <span className="text-sm font-semibold font-mono text-white uppercase tracking-wider">
          Expert System & XAI
        </span>
        {expert && (
          <span className="ml-auto text-xs font-mono text-dark-400 tabular-nums">
            {Math.round(expert.overall_confidence * 100)}% overall confidence
          </span>
        )}
      </div>

      {/* Connection / Face Status Warning */}
      {isWebcamRunning && !result?.face_detected && (
        <div className="flex items-center gap-2 text-xs font-mono text-amber-500 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2 animate-pulse">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
          No face detected — keeping last known metrics
        </div>
      )}

      {/* Active Alerts */}
      {activeAlerts.length > 0 ? (
        <div className="space-y-2">
          <span className="text-xs font-mono text-dark-500 uppercase tracking-wider">Active Alerts</span>
          {activeAlerts.map((alert, i) => (
            <AlertBadge key={i} alert={alert} />
          ))}
        </div>
      ) : (
        expert && expert.alerts.length === 0 && (
          <div className="flex items-center gap-2 text-xs font-mono text-brand-500 bg-brand-500/10 border border-brand-500/20 rounded-lg px-3 py-2">
            <CheckCircle size={12} />
            All systems nominal — no alerts
          </div>
        )
      )}

      {/* Scores */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-dark-800/60 border border-dark-600/40 rounded-lg p-3">
          <span className="text-dark-400 text-xs font-mono block">Fatigue Score</span>
          <span className={clsx(
            'text-xl font-bold font-mono tabular-nums',
            expert ? (
              expert.fatigue_score > 0.7 ? 'text-risk-critical' :
              expert.fatigue_score > 0.4 ? 'text-risk-medium' : 'text-brand-500'
            ) : 'text-dark-500'
          )}>
            {expert ? `${Math.round(expert.fatigue_score * 100)}%` : '--'}
          </span>
        </div>
        <div className="bg-dark-800/60 border border-dark-600/40 rounded-lg p-3">
          <span className="text-dark-400 text-xs font-mono block">Focus Score</span>
          <span className="text-xl font-bold font-mono text-metric-focus tabular-nums">
            {expert ? `${Math.round(expert.focus_score * 100)}%` : '--'}
          </span>
        </div>
      </div>

      {/* Explanations */}
      {expert?.explanations && expert.explanations.length > 0 && (
        <div className="space-y-2">
          <span className="text-xs font-mono text-dark-500 uppercase tracking-wider">
            Feature Explanations ({expert.explanations.length})
          </span>
          {expert.explanations.map((exp: any) => (
            <ExplanationCard key={exp.metric_name} explanation={exp} />
          ))}
        </div>
      )}
    </div>
  );
};
