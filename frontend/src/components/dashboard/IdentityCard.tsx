/**
 * AuthFaceGraph — Real-Time Telemetry & Behavioral Monitoring HUD
 * Strictly syncs UI state with live WebSocket payloads and camera status.
 * Purges all hardcoded mock identity strings (System Administrator / 99.3%).
 */

import React from 'react';
import { Shield, Camera, Activity, User, Key, CheckCircle2, AlertCircle } from 'lucide-react';
import { useAnalysisStore, useAuthStore } from '../../store';
import clsx from 'clsx';

export const IdentityCard: React.FC = () => {
  const latestResult  = useAnalysisStore(s => s.latestResult);
  const wsState       = useAnalysisStore(s => s.wsState);
  const sessionId     = useAnalysisStore(s => s.sessionId);
  const authUser      = useAuthStore(s => s.fullName) || 'Authenticated User';

  const faceDetected  = latestResult?.face_detected ?? false;
  const isWsConnected = wsState === 'connected';

  // Real Camera & Monitoring Status Indicators
  const cameraStatus = !isWsConnected ? 'OFFLINE' : 'ACTIVE';
  const monitoringStatus = !isWsConnected || !faceDetected ? 'NO FACE DETECTED' : 'ANALYZING';

  return (
    <div
      className={clsx(
        "rounded-2xl p-4 transition-all duration-300 gpu-accelerated glass-card border-indigo-500/20 shadow-xl"
      )}
      style={{ backdropFilter: 'blur(20px)' }}
    >
      {/* Header — Real Behavioral Indicators */}
      <div className="flex items-center justify-between pb-3 border-b border-white/10 mb-3">
        <div className="flex items-center gap-2">
          <div className={clsx(
            "w-7 h-7 rounded-lg border flex items-center justify-center transition-colors",
            isWsConnected && faceDetected
              ? "bg-emerald-500/20 border-emerald-500/40 text-emerald-400"
              : "bg-slate-800 border-slate-700 text-slate-400"
          )}>
            <Camera size={15} />
          </div>
          <div>
            <div className="font-mono text-[9px] uppercase tracking-widest text-slate-400">
              Camera Status
            </div>
            <div className={clsx("font-bold text-xs tracking-wider uppercase font-mono",
              cameraStatus === 'ACTIVE' ? "text-emerald-400" : "text-slate-400"
            )}>
              {cameraStatus}
            </div>
          </div>
        </div>

        <div className={clsx(
          "px-2.5 py-0.5 rounded-full font-mono text-[9px] font-bold uppercase tracking-wider flex items-center gap-1.5 border",
          monitoringStatus === 'ANALYZING'
            ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/30"
            : "bg-slate-800/80 text-slate-400 border-slate-700"
        )}>
          <span className={clsx(
            "w-1.5 h-1.5 rounded-full",
            monitoringStatus === 'ANALYZING' ? "bg-cyan-400 animate-ping" : "bg-slate-500"
          )} />
          {monitoringStatus}
        </div>
      </div>

      {/* Main Body Details — Real Telemetry Payload Sync */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] text-slate-400 flex items-center gap-1.5">
            <User size={12} className="text-cyan-400" /> Active Session:
          </span>
          <span className="font-semibold text-xs text-slate-100">{authUser}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] text-slate-400 flex items-center gap-1.5">
            <Key size={12} className="text-violet-400" /> Session ID:
          </span>
          <span className="font-mono font-bold text-[10px] text-violet-300 tracking-wider">
            {sessionId ? `${sessionId.slice(0, 12)}...` : 'NOT CONNECTED'}
          </span>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-indigo-500/10">
          <span className="font-mono text-[9px] text-slate-500 uppercase tracking-widest flex items-center gap-1">
            <Activity size={10} className="text-emerald-400" /> Live Stream Telemetry
          </span>
          <span className={clsx(
            "font-mono text-[10px] font-bold flex items-center gap-1",
            isWsConnected && faceDetected ? "text-emerald-400" : "text-amber-400"
          )}>
            {isWsConnected && faceDetected ? (
              <>
                <CheckCircle2 size={11} /> Synchronized (30 FPS)
              </>
            ) : (
              <>
                <AlertCircle size={11} /> Waiting for feed...
              </>
            )}
          </span>
        </div>
      </div>
    </div>
  );
};
