import React from 'react';
import type { DLAnalysisResult } from '../../types/analysis';
import { Brain, Cpu, ShieldAlert, CheckCircle2 } from 'lucide-react';

interface XAIExplanationPanelProps {
  dl: DLAnalysisResult;
}

const LANDMARK_LABELS: Record<number, string> = {
  4: "Nose Tip",
  152: "Chin",
  33: "R-Eye Outer Corner",
  133: "R-Eye Inner Corner",
  263: "L-Eye Outer Corner",
  362: "L-Eye Inner Corner",
  61: "Mouth Right Corner",
  291: "Mouth Left Corner",
  13: "Upper Lip Inner",
  14: "Lower Lip Inner",
  70: "R-Eyebrow Inner",
  336: "L-Eyebrow Inner",
};

export const XAIExplanationPanel: React.FC<XAIExplanationPanelProps> = ({ dl }) => {
  const explanations = dl.xai_explanations || [];
  const topLandmarks = dl.top_important_landmarks || [];

  return (
    <div className="bg-dark-800/40 border border-dark-600/50 rounded-2xl p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-dark-600/40 pb-3">
        <div className="flex items-center gap-2">
          <Brain size={18} className="text-violet-400" />
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
            Explainable AI (XAI) Engine
          </h3>
        </div>
        <span className="text-[10px] bg-violet-500/10 text-violet-400 border border-violet-500/20 px-2 py-0.5 rounded-full font-mono">
          Live Attribution
        </span>
      </div>

      {/* Explanations List */}
      {explanations.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-6 text-center text-dark-400">
          <Cpu size={24} className="mb-2 opacity-40" />
          <p className="text-xs">Waiting for GNN model inference attributions...</p>
        </div>
      ) : (
        explanations.map((exp, idx) => (
          <div key={idx} className="flex flex-col gap-4">
            {/* Plain English Summary Box */}
            <div className="bg-slate-950/45 border border-dark-600/30 rounded-xl p-3.5 font-mono text-xs text-slate-300 leading-relaxed shadow-inner">
              <div className="text-[10px] text-dark-400 uppercase mb-1.5 flex items-center gap-1.5">
                <CheckCircle2 size={12} className="text-emerald-400" />
                Inference Summary:
              </div>
              {exp.explanation_text || "No explanatory text generated."}
            </div>

            {/* Region Attributions */}
            {exp.attributions && exp.attributions.length > 0 && (
              <div className="flex flex-col gap-2.5">
                <span className="text-[10px] text-dark-300 font-mono uppercase tracking-wider">
                  GNN Region Attribution Weight
                </span>
                <div className="space-y-2">
                  {exp.attributions
                    .sort((a, b) => b.contribution - a.contribution)
                    .map((attr, attrIdx) => {
                      const percentage = Math.round(attr.contribution * 100);
                      const cleanName = attr.feature_name.replace('region_', '').toUpperCase();
                      
                      return (
                        <div key={attrIdx} className="space-y-1">
                          <div className="flex justify-between text-[11px] font-mono">
                            <span className="text-slate-300 capitalize">{cleanName}</span>
                            <span className="text-violet-400 font-bold">{percentage}%</span>
                          </div>
                          <div className="h-1.5 w-full bg-slate-950/50 rounded-full overflow-hidden border border-dark-600/20">
                            <div 
                              className="h-full bg-gradient-to-r from-violet-600 to-indigo-400 rounded-full transition-all duration-300"
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}
          </div>
        ))
      )}

      {/* Top Attributed Node Badges */}
      {topLandmarks.length > 0 && (
        <div className="flex flex-col gap-2 border-t border-dark-600/30 pt-3">
          <span className="text-[10px] text-dark-300 font-mono uppercase tracking-wider">
            Critical Nodes (Highest Activation)
          </span>
          <div className="flex flex-wrap gap-1.5 max-h-[110px] overflow-y-auto pr-1.5 scrollbar-thin scrollbar-thumb-dark-600 scrollbar-track-transparent">
            {topLandmarks.slice(0, 10).map((lmIdx, i) => {
              const label = LANDMARK_LABELS[lmIdx];
              return (
                <div 
                  key={lmIdx}
                  className="px-2 py-1 bg-slate-950/30 border border-dark-600/40 rounded-lg flex items-center gap-1.5 transition-all hover:border-violet-500/30"
                >
                  <span className="text-[9px] font-bold text-dark-400">#{i + 1}</span>
                  <span className="text-[10px] text-slate-300 font-semibold font-mono">
                    {label ? `${label} (${lmIdx})` : `Node ${lmIdx}`}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
