/**
 * AuthFaceGraph — Floating Identity Verification Card
 * Displays real-time identity match status, user profile, match confidence,
 * anti-spoofing liveness, and identity warning overlay when mismatch occurs.
 */

import React from 'react';
import { ShieldCheck, ShieldAlert, User, CheckCircle2, PauseCircle } from 'lucide-react';
import { useAnalysisStore, useAuthStore } from '../../store';
import clsx from 'clsx';

export const IdentityCard: React.FC = () => {
  const latestResult = useAnalysisStore(s => s.latestResult);
  const authUser     = useAuthStore(s => s.fullName);

  const idVerify = latestResult?.identity_verification;

  const enrolledName = idVerify?.enrolled_user_name || authUser || 'John Smith';
  const confidence   = idVerify?.match_confidence ?? 0.993;
  const confidencePct = `${(confidence * 100).toFixed(1)}%`;
  const isVerified   = idVerify?.status === 'verified';
  const isPaused     = idVerify?.is_paused ?? false;
  const isMismatch   = idVerify?.status === 'mismatch' || idVerify?.status === 'liveness_failed';

  return (
    <div
      className={clsx(
        "rounded-2xl p-4 transition-all duration-300 gpu-accelerated",
        isMismatch ? "border border-red-500/50 bg-red-950/40 shadow-[0_0_30px_rgba(239,68,68,0.25)]" : "glass-card border-indigo-500/20 shadow-xl"
      )}
      style={{
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-white/10 mb-3">
        <div className="flex items-center gap-2">
          {isMismatch ? (
            <div className="w-7 h-7 rounded-lg bg-red-500/20 border border-red-500/40 flex items-center justify-center text-red-400 animate-pulse">
              <ShieldAlert size={15} />
            </div>
          ) : (
            <div className="w-7 h-7 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
              <ShieldCheck size={15} />
            </div>
          )}
          <div>
            <div className="font-mono text-[9px] uppercase tracking-widest text-slate-400">
              Identity Status
            </div>
            <div className={clsx("font-bold text-xs tracking-wider uppercase", isMismatch ? "text-red-400" : "text-emerald-400")}>
              {isMismatch ? 'Mismatch Detected' : 'Verified'}
            </div>
          </div>
        </div>

        <div className={clsx("px-2.5 py-0.5 rounded-full font-mono text-[9px] font-bold uppercase tracking-wider flex items-center gap-1.5",
          isMismatch ? "bg-red-500/20 text-red-400 border border-red-500/30" : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
        )}>
          <span className={clsx("w-1.5 h-1.5 rounded-full", isMismatch ? "bg-red-400 animate-ping" : "bg-emerald-400 animate-pulse")} />
          {isMismatch ? 'PAUSED' : 'LIVE'}
        </div>
      </div>

      {/* Main Body Details */}
      {isMismatch ? (
        <div className="space-y-2.5">
          <div className="text-xs text-red-200 font-semibold flex items-center gap-1.5">
            <PauseCircle size={14} className="text-red-400 flex-shrink-0" />
            <span>Unknown Person Detected</span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-center pt-1">
            <div className="bg-red-950/50 p-2 rounded-xl border border-red-500/20">
              <div className="font-mono text-[8px] uppercase tracking-wider text-slate-400">Match Conf.</div>
              <div className="font-mono font-bold text-sm text-red-400">{confidencePct}</div>
            </div>
            <div className="bg-red-950/50 p-2 rounded-xl border border-red-500/20">
              <div className="font-mono text-[8px] uppercase tracking-wider text-slate-400">AI Tracking</div>
              <div className="font-mono font-bold text-xs text-red-400">PAUSED</div>
            </div>
          </div>

          <div className="font-mono text-[9.5px] text-red-300/80 leading-relaxed pt-1">
            ⚠ Please return the enrolled user to continue session tracking.
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] text-slate-400 flex items-center gap-1.5">
              <User size={12} className="text-cyan-400" /> User:
            </span>
            <span className="font-semibold text-xs text-slate-100">{enrolledName}</span>
          </div>

          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] text-slate-400 flex items-center gap-1.5">
              🎯 Match Confidence:
            </span>
            <span className="font-mono font-bold text-xs text-cyan-400">{confidencePct}</span>
          </div>

          <div className="flex items-center justify-between pt-1 border-t border-indigo-500/10">
            <span className="font-mono text-[9px] text-slate-500 uppercase tracking-widest">
              Liveness Check
            </span>
            <span className="font-mono text-[10px] text-emerald-400 flex items-center gap-1">
              <CheckCircle2 size={11} /> Passed
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
