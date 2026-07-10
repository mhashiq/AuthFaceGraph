import React from 'react';
import { EnsembleResult } from '../../types/analysis';

interface EnsemblePanelProps {
  ensemble: EnsembleResult;
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
  availableModels: string[];
}

export const EnsemblePanel: React.FC<EnsemblePanelProps> = ({
  ensemble,
  selectedModel,
  onSelectModel,
  availableModels,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl transition-all duration-300 hover:border-violet-500/50">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-500 animate-pulse" />
          Model Ensemble
        </h3>
        
        {/* Model Selector Dropdown */}
        <select
          value={selectedModel}
          onChange={(e) => onSelectModel(e.target.value)}
          className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-300 focus:outline-none focus:border-violet-500 transition-all font-mono"
        >
          <option value="ensemble">Ensemble Consensus</option>
          {availableModels.map((m) => (
            <option key={m} value={m}>
              {m.replace('_', ' ').toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Consensus display */}
        <div className="flex-1 bg-slate-950/40 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-center items-center text-center">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
            {selectedModel === 'ensemble' ? 'Consensus' : selectedModel.replace('_', ' ')} Emotion
          </span>
          <span className="text-3xl font-extrabold text-violet-400 capitalize mb-2">
            {selectedModel === 'ensemble' 
              ? ensemble.final_emotion 
              : (ensemble.model_predictions.find(p => p.model_id === selectedModel)?.emotion ?? 'unknown')
            }
          </span>
          <div className="flex items-center gap-2 mb-4">
            <span className="text-sm font-bold text-slate-200">
              {selectedModel === 'ensemble'
                ? (ensemble.confidence * 100).toFixed(0)
                : ((ensemble.model_predictions.find(p => p.model_id === selectedModel)?.confidence ?? 0) * 100).toFixed(0)
              }%
            </span>
            <span className="text-xs text-slate-400 font-medium">confidence</span>
          </div>

          <div className="w-full grid grid-cols-2 gap-4 border-t border-slate-800/80 pt-4">
            <div className="text-center">
              <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block">
                Disagreement
              </span>
              <span className="text-sm font-bold text-slate-300">
                {(ensemble.disagreement_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="text-center">
              <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block">
                Uncertainty
              </span>
              <span className="text-sm font-bold text-slate-300">
                {(ensemble.uncertainty * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {/* Member model list */}
        <div className="flex-[1.5] flex flex-col gap-3">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-0.5">
            Model Outputs
          </span>
          {ensemble.model_predictions.map((pred) => (
            <div
              key={pred.model_id}
              className="p-3 bg-slate-950/30 border border-slate-800/60 rounded-xl flex items-center justify-between"
            >
              <div>
                <span className="text-xs font-bold text-slate-300 block capitalize">
                  {pred.model_id.replace('_', ' ')}
                </span>
                <span className="text-[10px] text-slate-500 font-medium">
                  Latency: {pred.latency_ms.toFixed(1)}ms
                </span>
              </div>
              <div className="text-right">
                <span className="text-xs font-bold text-violet-400 block capitalize">
                  {pred.emotion}
                </span>
                <span className="text-[10px] text-slate-400 font-medium">
                  {(pred.confidence * 100).toFixed(0)}% conf
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
