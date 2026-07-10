import React from 'react';
import { useAuthStore, useAnalysisStore } from '../../store';
import { Shield, User, Cpu, Activity, Clock, LogOut, FileText, Database } from 'lucide-react';
import clsx from 'clsx';

interface LeftSidebarProps {
  onProfileClick: () => void;
  onLogout: () => void;
}

export const LeftSidebar: React.FC<LeftSidebarProps> = ({ onProfileClick, onLogout }) => {
  const { fullName, role, email } = useAuthStore();
  const wsState = useAnalysisStore(s => s.wsState);
  const result  = useAnalysisStore(s => s.latestResult);

  const statusConfigs = {
    connected:    { label: 'Live Link',      cls: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' },
    connecting:   { label: 'Connecting',     cls: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30 animate-pulse' },
    disconnected: { label: 'Offline Mode',   cls: 'bg-slate-500/10 text-slate-400 border-slate-500/30' },
    error:        { label: 'Link Failure',   cls: 'bg-rose-500/10 text-rose-400 border-rose-500/30' },
  };

  const status = statusConfigs[wsState] || statusConfigs.disconnected;

  return (
    <aside className="bg-dark-900 border border-dark-600/60 rounded-2xl p-5 flex flex-col gap-6 shadow-2xl h-full">
      {/* Brand Header */}
      <div className="flex flex-col gap-1 border-b border-dark-700/60 pb-4">
        <h2 className="text-sm font-extrabold tracking-widest text-white uppercase font-mono flex items-center gap-2">
          <Shield size={16} className="text-violet-400" />
          AuthFaceGraph AI
        </h2>
        <span className="text-[10px] text-dark-400 font-mono tracking-wider">
          SYSTEM VER 2.0.0 (ONNX-ACCL)
        </span>
      </div>

      {/* User Profile Summary */}
      <div 
        onClick={onProfileClick}
        className="flex items-center gap-3 p-3 bg-slate-950/40 border border-dark-600/40 rounded-xl hover:border-violet-500/30 transition-all cursor-pointer group"
      >
        <div className="w-10 h-10 rounded-full bg-violet-600/20 border border-violet-500/40 flex items-center justify-center group-hover:bg-violet-600/30 transition-colors">
          <User size={18} className="text-violet-400" />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-xs font-bold text-slate-200 truncate group-hover:text-white transition-colors">
            {fullName || 'Guest Operator'}
          </h4>
          <span className="text-[10px] text-violet-400 font-mono capitalize">
            {role || 'Viewer'}
          </span>
        </div>
      </div>

      {/* Status Indicators */}
      <div className="space-y-3 font-mono text-[10px]">
        <span className="text-dark-400 uppercase tracking-wider block border-b border-dark-700/30 pb-1.5 mb-2">
          Telemetric Status
        </span>
        <div className="flex justify-between items-center">
          <span className="text-slate-400">SESSION LINK:</span>
          <span className={clsx('px-2 py-0.5 rounded border text-[9px] font-bold uppercase', status.cls)}>
            {status.label}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-slate-400">HARDWARE TYPE:</span>
          <span className="text-white font-semibold flex items-center gap-1">
            <Cpu size={11} className="text-violet-400" /> CPU ENGINE
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-slate-400">DBMS ENGINE:</span>
          <span className="text-slate-200 font-semibold flex items-center gap-1">
            <Database size={11} className="text-violet-400" /> SQLITE
          </span>
        </div>
      </div>

      {/* Real-Time Performance */}
      <div className="space-y-3 font-mono text-[10px] bg-slate-950/20 border border-dark-600/30 p-3 rounded-xl">
        <span className="text-dark-300 uppercase block font-semibold mb-1">
          Performance Indicators
        </span>
        <div className="flex justify-between items-center">
          <span className="text-dark-400">FRAME SPEED:</span>
          <span className="text-emerald-400 font-bold flex items-center gap-1">
            <Activity size={11} /> {result ? `${result.fps.toFixed(1)} FPS` : '—'}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-dark-400">INFERENCE LATENCY:</span>
          <span className="text-violet-400 font-bold flex items-center gap-1">
            <Clock size={11} /> {result ? `${result.inference_time_ms.toFixed(1)} ms` : '—'}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-dark-400">LANDMARKS PARSED:</span>
          <span className="text-slate-200 font-bold">
            {result?.face_detected ? '478 / 478' : '0 / 478'}
          </span>
        </div>
      </div>

      {/* Navigation & Controls */}
      <div className="flex-1 flex flex-col justify-end gap-2.5">
        <button
          onClick={onProfileClick}
          className="w-full py-2 px-3 bg-slate-950/40 hover:bg-slate-950/80 border border-dark-600/50 hover:border-violet-500/30 rounded-lg text-dark-300 hover:text-white transition-all text-xs font-mono font-medium flex items-center gap-2"
        >
          <FileText size={14} className="text-violet-400" />
          Session Records
        </button>
        
        <button
          onClick={onLogout}
          className="w-full py-2 px-3 bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 rounded-lg text-rose-400 hover:text-rose-300 transition-all text-xs font-mono font-medium flex items-center gap-2"
        >
          <LogOut size={14} />
          Operator Exit
        </button>
      </div>
    </aside>
  );
};
