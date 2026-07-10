import React, { useEffect, useState } from 'react';
import { X, RefreshCw, User, Calendar, Clock, Eye, AlertTriangle, ShieldCheck, LogOut, Loader2 } from 'lucide-react';
import { useAuthStore, useAnalysisStore } from '../../store';

interface ProfileHistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

interface SessionRecord {
  session_id: string;
  started_at: string;
  ended_at: string | null;
  status: string;
  total_frames: number;
  total_blinks: number;
  avg_ear: number;
  avg_head_yaw: number;
  avg_head_pitch: number;
  dominant_attention_state: string;
  face_quality_score: number;
}

export const ProfileHistoryDrawer: React.FC<ProfileHistoryDrawerProps> = ({ isOpen, onClose }) => {
  const { fullName, email, role, orgId, accessToken, clearAuth } = useAuthStore();
  const { consentGranted, setConsentGranted, sessionId } = useAnalysisStore();
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'profile' | 'history'>('history');

  const fetchHistory = async () => {
    if (!accessToken) {
      setError('Not authenticated — please log in again.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/sessions/?limit=20', {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`HTTP ${response.status}: ${body || response.statusText}`);
      }

      const data = await response.json();
      console.log('[ProfileDrawer] sessions fetched:', data);
      setSessions(Array.isArray(data) ? data : []);
    } catch (err: any) {
      console.error('[ProfileDrawer] fetch error:', err);
      setError(err.message || 'An error occurred while loading history.');
    } finally {
      setLoading(false);
    }
  };

  // Fetch when drawer opens OR when user switches to history tab
  useEffect(() => {
    if (isOpen && activeTab === 'history') {
      fetchHistory();
    }
  }, [isOpen, activeTab]);

  const handleLogout = () => {
    // Clear user auth store
    clearAuth();
    // Redirect to login page (which is ConsentPage at "/")
    window.location.href = '/';
  };

  const handleRevokeConsent = async () => {
    if (!sessionId || !accessToken) return;
    if (window.confirm('Are you sure you want to revoke consent? This will stop active webcam analysis immediately.')) {
      try {
        await fetch(`/api/consent/${sessionId}`, {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${accessToken}`,
          },
        });
        setConsentGranted(false);
        onClose();
        window.location.href = '/';
      } catch (err) {
        console.error('Failed to revoke consent', err);
      }
    }
  };

  if (!isOpen) return null;

  // Format date utility
  const formatDate = (dateStr: string) => {
    try {
      const d = new Date(dateStr);
      return d.toLocaleString(undefined, {
        dateStyle: 'medium',
        timeStyle: 'short',
      });
    } catch {
      return dateStr;
    }
  };

  // Helper to compute session duration
  const getDurationStr = (session: SessionRecord) => {
    if (!session.ended_at) return 'Active';
    try {
      // Normalise timestamps: if no timezone offset present, treat as UTC by appending 'Z'
      const normalise = (ts: string) =>
        /Z$|[+-]\d{2}:\d{2}$/.test(ts) ? ts : ts + 'Z';

      const start = new Date(normalise(session.started_at)).getTime();
      const end   = new Date(normalise(session.ended_at)).getTime();
      const totalSec = Math.floor((end - start) / 1000);

      if (totalSec <= 0) return '< 1s';
      if (totalSec < 60) return `${totalSec}s`;
      const min = Math.floor(totalSec / 60);
      const sec = totalSec % 60;
      return sec > 0 ? `${min}m ${sec}s` : `${min}m`;
    } catch {
      return 'N/A';
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-dark-950/80 backdrop-blur-sm cursor-pointer" 
        onClick={onClose}
      />

      {/* Drawer Content */}
      <div className="relative w-full max-w-md h-full bg-dark-900 border-l border-dark-700/50 shadow-2xl flex flex-col z-10 animate-in slide-in-from-right duration-200">
        
        {/* Drawer Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-dark-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-brand-500/10 border border-brand-500/30 flex items-center justify-center">
              <User className="text-brand-500" size={20} />
            </div>
            <div>
              <h2 className="text-white font-bold text-base leading-none">Account Settings</h2>
              <p className="text-dark-400 text-xs mt-1 font-mono">{email}</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-dark-800 text-dark-400 hover:text-white transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Navigation Tabs */}
        <div className="flex border-b border-dark-800 px-6">
          <button
            onClick={() => setActiveTab('profile')}
            className={`py-3 text-sm font-medium border-b-2 px-2 transition-all ${
              activeTab === 'profile'
                ? 'border-brand-500 text-brand-500 text-glow-brand font-semibold'
                : 'border-transparent text-dark-400 hover:text-white'
            }`}
          >
            User Profile
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`py-3 text-sm font-medium border-b-2 px-2 ml-6 transition-all ${
              activeTab === 'history'
                ? 'border-brand-500 text-brand-500 text-glow-brand font-semibold'
                : 'border-transparent text-dark-400 hover:text-white'
            }`}
          >
            Session History
          </button>
        </div>

        {/* Tab Contents */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          
          {activeTab === 'profile' && (
            <div className="space-y-6">
              
              {/* User Info Details Card */}
              <div className="glass-card rounded-xl p-5 space-y-4">
                <h3 className="text-white font-semibold text-sm font-mono border-b border-dark-800 pb-2">Profile Details</h3>
                
                <div className="grid grid-cols-3 gap-y-3 text-xs">
                  <div className="text-dark-400 font-mono">Full Name:</div>
                  <div className="text-white font-medium col-span-2">{fullName || 'User'}</div>
                  
                  <div className="text-dark-400 font-mono">Role:</div>
                  <div className="text-white font-semibold capitalize col-span-2">
                    <span className="px-2 py-0.5 rounded bg-brand-500/10 text-brand-500 border border-brand-500/20 text-[10px] tracking-wide">
                      {role}
                    </span>
                  </div>

                  <div className="text-dark-400 font-mono">Org ID:</div>
                  <div className="text-dark-300 font-mono text-[10px] select-all col-span-2 truncate">{orgId}</div>
                  
                  <div className="text-dark-400 font-mono">Consent Status:</div>
                  <div className="text-brand-500 font-medium flex items-center gap-1 col-span-2">
                    <ShieldCheck size={14} /> Granted (GDPR compliant)
                  </div>
                </div>
              </div>

              {/* GDPR Rights & Actions */}
              <div className="glass-card rounded-xl p-5 space-y-4">
                <h3 className="text-white font-semibold text-sm font-mono border-b border-dark-800 pb-2">Privacy & GDPR Rights</h3>
                <p className="text-dark-400 text-xs leading-relaxed">
                  As a GDPR-protected user, you have the right to withdraw your consent at any time. Revoking consent deletes your active analysis token and terminates any active analysis session immediately.
                </p>
                <div className="pt-2">
                  <button
                    onClick={handleRevokeConsent}
                    className="w-full py-2.5 px-4 text-xs font-semibold text-red-500 hover:text-white bg-red-500/10 hover:bg-red-600 border border-red-500/20 hover:border-red-600 rounded-lg transition-all flex items-center justify-center gap-2"
                  >
                    <AlertTriangle size={14} />
                    Revoke Consent & Stop Session
                  </button>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-4 border-t border-dark-800">
                <button
                  onClick={handleLogout}
                  className="w-full py-3 px-4 text-sm font-semibold text-white bg-dark-800 hover:bg-dark-700 border border-dark-700 hover:border-dark-600 rounded-xl transition-all flex items-center justify-center gap-2"
                >
                  <LogOut size={16} />
                  Sign Out of Account
                </button>
              </div>

            </div>
          )}

          {activeTab === 'history' && (
            <div className="space-y-4">
              
              {/* Header and Refresh */}
              <div className="flex items-center justify-between">
                <span className="text-dark-400 text-xs font-mono">Showing last 20 sessions</span>
                <button 
                  onClick={fetchHistory}
                  disabled={loading}
                  className="text-brand-500 hover:text-brand-400 text-xs font-medium font-mono flex items-center gap-1 disabled:opacity-50"
                >
                  {loading ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                  Refresh
                </button>
              </div>

              {/* History List */}
              {loading && sessions.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 text-dark-500">
                  <Loader2 size={24} className="animate-spin text-brand-500" />
                  <p className="text-xs font-mono mt-3">Fetching records...</p>
                </div>
              ) : error ? (
                <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-xs font-mono">
                  {error}
                </div>
              ) : sessions.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 text-dark-500 text-center">
                  <Calendar size={32} className="text-dark-700 mb-3" />
                  <p className="text-xs font-mono">No analysis history found.</p>
                  <p className="text-[10px] text-dark-600 mt-1 max-w-[200px]">Completed sessions will be saved automatically upon disconnection.</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {sessions.map((session) => (
                    <div 
                      key={session.session_id} 
                      className="glass-card rounded-xl p-4 border border-dark-700/40 hover:border-dark-600/60 transition-all space-y-3"
                    >
                      {/* Session Top Bar */}
                      <div className="flex items-center justify-between border-b border-dark-800 pb-2">
                        <div className="flex items-center gap-1.5 text-white font-semibold text-xs">
                          <Calendar size={12} className="text-brand-500" />
                          <span>{formatDate(session.started_at)}</span>
                        </div>
                        <span className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded border ${
                          session.dominant_attention_state === 'focused'
                            ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
                            : session.dominant_attention_state === 'distracted'
                            ? 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'
                            : 'bg-red-500/10 text-red-500 border-red-500/20'
                        }`}>
                          {session.dominant_attention_state || 'unknown'}
                        </span>
                      </div>

                      {/* Session Parameters */}
                      <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-[11px] font-mono">
                        <div className="flex items-center gap-1 text-dark-400">
                          <Clock size={11} /> Duration:
                        </div>
                        <div className="text-white text-right">{getDurationStr(session)}</div>

                        <div className="flex items-center gap-1 text-dark-400">
                          <Eye size={11} /> Total Blinks:
                        </div>
                        <div className="text-white text-right">{session.total_blinks}</div>

                        <div className="flex items-center gap-1 text-dark-400">
                          <ShieldCheck size={11} /> Avg EAR:
                        </div>
                        <div className="text-white text-right">{(session.avg_ear || 0).toFixed(3)}</div>

                        <div className="flex items-center gap-1 text-dark-400">
                          <AlertTriangle size={11} /> Avg Pose Yaw/Pitch:
                        </div>
                        <div className="text-white text-right">
                          {(session.avg_head_yaw || 0).toFixed(1)}° / {(session.avg_head_pitch || 0).toFixed(1)}°
                        </div>
                      </div>

                      {/* Score badges */}
                      <div className="flex items-center gap-2 pt-1">
                        <div className="flex-1 bg-dark-950 rounded p-1.5 text-center">
                          <div className="text-[9px] text-dark-500 uppercase font-mono">Avg Quality</div>
                          <div className="text-[11px] text-glow-brand text-brand-500 font-bold font-mono">
                            {Math.round((session.face_quality_score || 0) * 100)}%
                          </div>
                        </div>
                        <div className="flex-1 bg-dark-950 rounded p-1.5 text-center">
                          <div className="text-[9px] text-dark-500 uppercase font-mono">Frames</div>
                          <div className="text-[11px] text-white font-bold font-mono">
                            {session.total_frames}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

        </div>

      </div>
    </div>
  );
};
