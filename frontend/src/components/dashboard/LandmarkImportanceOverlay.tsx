import React from 'react';

interface LandmarkImportanceOverlayProps {
  topLandmarks: number[];
}

export const LandmarkImportanceOverlay: React.FC<LandmarkImportanceOverlayProps> = ({
  topLandmarks,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl transition-all duration-300 hover:border-violet-500/50">
      <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-violet-400 animate-pulse" />
        GNN Node Importance (Top 20 Landmarks)
      </h3>
      {topLandmarks.length === 0 ? (
        <p className="text-slate-500 text-sm text-center py-8">GNNExplainer node importance not computed.</p>
      ) : (
        <div className="flex flex-wrap gap-2 max-h-[180px] overflow-y-auto pr-2 custom-scrollbar">
          {topLandmarks.map((lmIdx, i) => (
            <div
              key={lmIdx}
              className="px-3 py-1.5 bg-slate-950/40 border border-slate-800/80 rounded-lg flex items-center gap-2 transition-all hover:border-violet-500/40"
            >
              <span className="text-[10px] font-bold text-slate-500">#{i + 1}</span>
              <span className="text-xs font-semibold text-violet-400 font-mono">Index {lmIdx}</span>
            </div>
          ))}
        </div>
      )}
      <p className="text-xs text-slate-500 mt-4 leading-relaxed">
        GNNExplainer attributions identify which coordinates are most sensitive to graph-level perturbation under the active Graph Attention network.
      </p>
    </div>
  );
};
