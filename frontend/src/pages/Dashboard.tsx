/**
 * AuthBrain AI Face Analysis Engine
 * Main Dashboard Page
 *
 * Production-grade real-time analysis dashboard with:
 * - Live webcam with face mesh overlay
 * - Real-time metrics grid
 * - EAR/Head Pose/Fatigue charts
 * - Expert System XAI panel
 * - System logs
 */

import React, { useState } from 'react';
import { Cpu, Activity, Brain, Shield } from 'lucide-react';
import { CameraFeed } from '../components/webcam/CameraFeed';
import { MetricsPanel } from '../components/dashboard/MetricsPanel';
import { ExpertSystemPanel } from '../components/dashboard/ExpertSystemPanel';
import { SystemLogs } from '../components/dashboard/SystemLogs';
import { EARChart, HeadPoseChart, FatigueChart } from '../components/charts/EARChart';
import { ProfileHistoryDrawer } from '../components/dashboard/ProfileHistoryDrawer';
import { useAnalysisStore, useAuthStore } from '../store';

// Deep Learning Platform Components
import { EmotionRadarChart } from '../components/dashboard/EmotionRadarChart';
import { ActionUnitsPanel } from '../components/dashboard/ActionUnitsPanel';
import { EnsemblePanel } from '../components/dashboard/EnsemblePanel';
import { XAIExplanationPanel } from '../components/dashboard/XAIExplanationPanel';
import { EmotionTimelineChart } from '../components/dashboard/EmotionTimelineChart';

// ── Status Indicator ──────────────────────────────────────────────────────────

const StatusDot: React.FC<{ active: boolean }> = ({ active }) => (
  <span className={`inline-block w-2 h-2 rounded-full ${active ? 'bg-brand-500 animate-pulse' : 'bg-dark-500'}`} />
);

// ── Dashboard Header ──────────────────────────────────────────────────────────

interface DashboardHeaderProps {
  onProfileClick: () => void;
}

const DashboardHeader: React.FC<DashboardHeaderProps> = ({ onProfileClick }) => {
  const { fullName, role, email } = useAuthStore();
  const wsState = useAnalysisStore(s => s.wsState);
  const result  = useAnalysisStore(s => s.latestResult);

  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-dark-700/50 bg-dark-900/80 backdrop-blur-md sticky top-0 z-40">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <StatusDot active={wsState === 'connected'} />
        <div>
          <h1 className="text-white font-bold text-sm leading-tight tracking-tight uppercase">AuthFaceGraph</h1>
          <p className="text-dark-400 text-xs font-mono">Face Analysis Engine v1.0</p>
        </div>
      </div>

      {/* Live stats */}
      {result && (
        <div className="hidden md:flex items-center gap-6 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <Cpu size={12} className="text-dark-500" />
            <span className="text-dark-400">{result.fps.toFixed(1)} FPS</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Activity size={12} className="text-dark-500" />
            <span className="text-dark-400">{result.inference_time_ms.toFixed(1)}ms</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Shield size={12} className="text-dark-500" />
            <span className="text-dark-400">{Math.round((result.model_confidence ?? 0) * 100)}% conf</span>
          </div>
        </div>
      )}

      {/* User info */}
      <button 
        onClick={onProfileClick}
        className="flex items-center gap-3 hover:bg-dark-800/40 p-1 px-2 rounded-xl transition-all border border-transparent hover:border-dark-700/50 text-left cursor-pointer group"
      >
        <div className="text-right hidden sm:block">
          <p className="text-white text-sm font-semibold leading-tight group-hover:text-brand-400 transition-colors">{fullName || email}</p>
          <p className="text-dark-400 text-xs font-mono capitalize">{role}</p>
        </div>
        <div className="w-9 h-9 rounded-full bg-dark-700 group-hover:bg-brand-500/10 border border-dark-600 group-hover:border-brand-500/30 flex items-center justify-center transition-all">
          <span className="text-brand-500 text-sm font-bold font-mono">
            {(fullName || email || 'U').charAt(0).toUpperCase()}
          </span>
        </div>
      </button>
    </header>
  );
};


// ── Main Dashboard ─────────────────────────────────────────────────────────────

export const Dashboard: React.FC = () => {
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('ensemble');
  const latestResult = useAnalysisStore((s) => s.latestResult);
  const emotionHistory = useAnalysisStore((s) => s.emotionHistory);

  const dl = latestResult?.deep_learning;
  const dlEnabled = dl?.dl_enabled ?? false;

  // Resolve active probabilities based on selected model
  let probabilities = dl?.emotion_ensemble?.probabilities ?? {};
  if (selectedModel !== 'ensemble' && dl?.emotion_ensemble?.model_predictions) {
    const pred = dl.emotion_ensemble.model_predictions.find((p) => p.model_id === selectedModel);
    if (pred) {
      probabilities = pred.probabilities;
    }
  } else if (selectedModel === 'gnn_gat' && dl?.gnn_prediction) {
    probabilities = dl.gnn_prediction.probabilities;
  }

  // Get list of active models used in this frame
  const availableModels = dl?.models_used ?? [];

  return (
    <div className="min-h-screen bg-dark-950 text-white flex flex-col" style={{ backgroundImage: 'var(--tw-bg-grid-pattern)' }}>
      <DashboardHeader onProfileClick={() => setIsProfileOpen(true)} />

      <main className="flex-1 p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-5 max-w-[1600px] mx-auto w-full">

        {/* ── Left Column: Camera + Expert System ───────────────────────────── */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          {/* Camera Feed */}
          <CameraFeed className="flex-shrink-0" />

          {/* Expert System Panel */}
          <div className="flex-1">
            <ExpertSystemPanel />
          </div>

          {/* Explainable AI (XAI) Panel */}
          {dlEnabled && dl && (
            <XAIExplanationPanel dl={dl} />
          )}
        </div>

        {/* ── Center Column: Charts ──────────────────────────────────────────── */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          <EARChart />
          <HeadPoseChart />
          <FatigueChart />

          {/* Emotion Timeline Chart */}
          {dlEnabled && (
            <EmotionTimelineChart data={emotionHistory} />
          )}

          <SystemLogs />
        </div>

        {/* ── Right Column: Metrics ──────────────────────────────────────────── */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          <MetricsPanel />

          {/* Emotion Ensemble Panel */}
          {dlEnabled && dl?.emotion_ensemble && (
            <EnsemblePanel
              ensemble={dl.emotion_ensemble}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
              availableModels={availableModels}
            />
          )}

          {/* Emotion Radar Chart */}
          {dlEnabled && probabilities && (
            <EmotionRadarChart probabilities={probabilities} />
          )}

          {/* FACS Action Units Panel */}
          {dlEnabled && dl?.action_units && (
            <ActionUnitsPanel actionUnits={dl.action_units} />
          )}
        </div>

      </main>

      {/* Footer */}
      <footer className="px-6 py-4 border-t border-dark-800/40 flex flex-col sm:flex-row items-center justify-between text-xs font-mono text-dark-500 gap-2">
        <div className="flex items-center gap-1.5">
          <span>© {new Date().getFullYear()}</span>
          <span className="text-white font-medium">Developed and Researched by AuthFaceGraph</span>
          <span>·</span>
          <a 
            href="https://www.authfacegraph.io" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-violet-400 hover:text-violet-300 hover:underline transition-colors font-semibold"
          >
            www.authfacegraph.io
          </a>
        </div>
        <span>Consent-based · GDPR Compliant · No Biometric Data Stored</span>
      </footer>

      {/* Profile & History Drawer — rendered at root level to sit above everything */}
      <ProfileHistoryDrawer
        isOpen={isProfileOpen}
        onClose={() => setIsProfileOpen(false)}
      />
    </div>
  );
};
