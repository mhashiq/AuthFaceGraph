import React from 'react';
import { ActionUnit } from '../../types/analysis';

interface ActionUnitsPanelProps {
  actionUnits: ActionUnit[];
}

export const ActionUnitsPanel: React.FC<ActionUnitsPanelProps> = ({ actionUnits }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl transition-all duration-300 hover:border-violet-500/50">
      <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
        FACS Action Units (AUs)
      </h3>
      {actionUnits.length === 0 ? (
        <p className="text-slate-500 text-sm text-center py-8">No Action Unit estimates available</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[260px] overflow-y-auto pr-2 custom-scrollbar">
          {actionUnits.map((au) => (
            <div
              key={au.au_id}
              className={`p-3 rounded-xl border transition-all duration-250 ${
                au.present
                  ? 'bg-emerald-950/20 border-emerald-500/30'
                  : 'bg-slate-950/40 border-slate-800/80'
              }`}
            >
              <div className="flex justify-between items-center mb-1.5">
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-bold px-2 py-0.5 rounded-md ${
                    au.present ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-400'
                  }`}>
                    {au.au_id}
                  </span>
                  <span className="text-xs font-medium text-slate-300 truncate max-w-[120px]">
                    {au.name}
                  </span>
                </div>
                <span className="text-xs text-slate-400 font-semibold">
                  Int: {au.intensity.toFixed(1)}/5.0
                </span>
              </div>
              {/* Intensity progress bar */}
              <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    au.present ? 'bg-emerald-500' : 'bg-slate-600'
                  }`}
                  style={{ width: `${(au.intensity / 5.0) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
