/**
 * AuthFaceGraph — Premium Left Sidebar
 * 5-section navigation with animated active state
 */

import React from 'react';
import { useAuthStore, useAnalysisStore } from '../../store';
import {
  Shield, LayoutDashboard, BarChart3, Users,
  Activity, Settings, LogOut, ChevronRight, Cpu
} from 'lucide-react';
import { StatusDot } from '../ui';
import clsx from 'clsx';

type Section = 'dashboard' | 'analytics' | 'users' | 'activity' | 'settings';

interface LeftSidebarProps {
  activeSection: Section;
  onSectionChange: (s: Section) => void;
  onProfileClick: () => void;
  onLogout: () => void;
}

const NAV_ITEMS: Array<{ id: Section; label: string; icon: React.ReactNode; description: string }> = [
  { id: 'dashboard',  label: 'Dashboard',   icon: <LayoutDashboard size={17} />, description: 'Live analysis cockpit' },
  { id: 'analytics',  label: 'Analytics',   icon: <BarChart3 size={17} />,       description: 'Deep metrics & insights' },
  { id: 'users',      label: 'Users',        icon: <Users size={17} />,           description: 'Identity management' },
  { id: 'activity',   label: 'Monitoring',  icon: <Activity size={17} />,        description: 'Real-time activity feed' },
  { id: 'settings',   label: 'Settings',    icon: <Settings size={17} />,        description: 'System configuration' },
];

export const LeftSidebar: React.FC<LeftSidebarProps> = ({
  activeSection, onSectionChange, onProfileClick, onLogout,
}) => {
  const { fullName, role, email } = useAuthStore();
  const wsState = useAnalysisStore(s => s.wsState);
  const fps     = useAnalysisStore(s => s.latestResult?.fps ?? 0);

  const wsStatusMap: Record<string, 'online' | 'offline' | 'warning' | 'error'> = {
    connected:    'online',
    connecting:   'warning',
    disconnected: 'offline',
    error:        'error',
  };
  const wsLabel: Record<string, string> = {
    connected:    'Live Link',
    connecting:   'Connecting',
    disconnected: 'Offline',
    error:        'Link Error',
  };

  return (
    <aside
      className="flex flex-col h-full rounded-2xl py-5 px-4 gap-5 gpu-accelerated"
      style={{
        background: 'rgba(7,13,26,0.8)',
        backdropFilter: 'blur(24px)',
        border: '1px solid rgba(79,70,229,0.18)',
        boxShadow: '0 8px 48px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04)',
      }}
    >
      {/* ── BRAND ─────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 pb-4 border-b border-indigo-500/10">
        <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{
            background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
            boxShadow: '0 0 20px rgba(139,92,246,0.5)',
          }}>
          <Shield size={17} className="text-white" />
        </div>
        <div className="min-w-0">
          <div className="font-display font-bold text-xs tracking-widest text-gradient-brand uppercase">
            AuthFaceGraph
          </div>
          <div className="font-mono text-[9px] text-slate-500 tracking-wider mt-0.5">
            v2.0.0 · ONNX ENGINE
          </div>
        </div>
      </div>

      {/* ── USER PROFILE ──────────────────────────────────────── */}
      <div
        onClick={onProfileClick}
        className="flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all duration-200 group"
        style={{ background: 'rgba(79,70,229,0.06)', border: '1px solid rgba(79,70,229,0.12)' }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLElement).style.background = 'rgba(79,70,229,0.12)';
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(139,92,246,0.3)';
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLElement).style.background = 'rgba(79,70,229,0.06)';
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(79,70,229,0.12)';
        }}
      >
        {/* Avatar */}
        <div className="relative w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #4f46e5, #7c3aed)', boxShadow: '0 0 12px rgba(139,92,246,0.4)' }}>
          <span className="text-sm font-bold text-white">
            {(fullName || 'G').charAt(0).toUpperCase()}
          </span>
          {/* Online indicator */}
          <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-2"
            style={{ background: '#10b981', borderColor: '#070d1a' }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs font-semibold text-slate-200 truncate group-hover:text-white transition-colors">
            {fullName || 'Guest Operator'}
          </div>
          <div className="font-mono text-[9px] capitalize" style={{ color: '#8b5cf6' }}>
            {role || 'Viewer'} · {email?.split('@')[0] || 'unknown'}
          </div>
        </div>
        <ChevronRight size={13} className="text-slate-500 group-hover:text-violet-400 transition-colors flex-shrink-0" />
      </div>

      {/* ── NAVIGATION ────────────────────────────────────────── */}
      <nav className="flex-1 flex flex-col gap-1">
        {NAV_ITEMS.map(item => {
          const isActive = activeSection === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSectionChange(item.id)}
              className={clsx(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-all duration-200 group',
                'border border-transparent',
              )}
              style={isActive ? {
                background: 'linear-gradient(135deg, rgba(79,70,229,0.2), rgba(139,92,246,0.1))',
                borderColor: 'rgba(0,212,255,0.25)',
                boxShadow: '0 0 16px rgba(0,212,255,0.08)',
              } : {}}
              onMouseEnter={e => {
                if (!isActive) {
                  const el = e.currentTarget as HTMLElement;
                  el.style.background = 'rgba(79,70,229,0.08)';
                  el.style.borderColor = 'rgba(79,70,229,0.2)';
                }
              }}
              onMouseLeave={e => {
                if (!isActive) {
                  const el = e.currentTarget as HTMLElement;
                  el.style.background = 'transparent';
                  el.style.borderColor = 'transparent';
                }
              }}
            >
              {/* Active line */}
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-r-full"
                  style={{ background: 'var(--cyan)', boxShadow: '0 0 8px var(--cyan)' }} />
              )}
              <span style={{ color: isActive ? '#00d4ff' : '#5d7399' }}
                className="flex-shrink-0 transition-colors group-hover:text-slate-300">
                {item.icon}
              </span>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold"
                  style={{ color: isActive ? '#f0f4ff' : '#8fa3cc' }}>
                  {item.label}
                </div>
                <div className="font-mono text-[9px] mt-0.5 truncate"
                  style={{ color: isActive ? 'rgba(0,212,255,0.6)' : '#3a4f70' }}>
                  {item.description}
                </div>
              </div>
              {isActive && (
                <div className="w-1.5 h-1.5 rounded-full flex-shrink-0 animate-pulse"
                  style={{ background: '#00d4ff', boxShadow: '0 0 6px #00d4ff' }} />
              )}
            </button>
          );
        })}
      </nav>

      {/* ── SYSTEM STATUS ─────────────────────────────────────── */}
      <div className="rounded-xl p-3 space-y-2"
        style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(79,70,229,0.1)' }}>
        <div className="font-mono text-[9px] uppercase tracking-widest text-slate-500 mb-2">System Status</div>
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] text-slate-400">WS Stream</span>
          <StatusDot status={wsStatusMap[wsState] || 'offline'} label={wsLabel[wsState] || 'Unknown'} />
        </div>
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] text-slate-400">FPS</span>
          <span className="font-mono text-[10px]" style={{ color: fps > 20 ? '#10b981' : '#f59e0b' }}>
            {fps.toFixed(0)} FPS
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] text-slate-400">Engine</span>
          <span className="font-mono text-[10px] flex items-center gap-1 text-violet-400">
            <Cpu size={9} /> ONNX-CPU
          </span>
        </div>
      </div>

      {/* ── LOGOUT ────────────────────────────────────────────── */}
      <button
        onClick={onLogout}
        className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl w-full transition-all duration-200 group"
        style={{ border: '1px solid rgba(239,68,68,0.1)' }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLElement).style.background = 'rgba(239,68,68,0.08)';
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(239,68,68,0.25)';
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLElement).style.background = 'transparent';
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(239,68,68,0.1)';
        }}
      >
        <LogOut size={15} className="text-red-400/60 group-hover:text-red-400 transition-colors" />
        <span className="text-xs font-mono text-red-400/60 group-hover:text-red-400 transition-colors tracking-wider">
          End Session
        </span>
      </button>
    </aside>
  );
};
