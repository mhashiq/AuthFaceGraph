/**
 * AuthFaceGraph AI
 * Main Dashboard Page
 *
 * Production-grade real-time analysis dashboard with a world-class
 * enterprise layout resembling Microsoft Build / NVIDIA GTC cockpit tools.
 */

import React, { useState } from 'react';
import { LeftSidebar } from '../components/dashboard/LeftSidebar';
import { TopBar } from '../components/dashboard/TopBar';
import { PrimaryAIVisualizer } from '../components/webcam/PrimaryAIVisualizer';
import { RightPanel } from '../components/dashboard/RightPanel';
import { AnalyticsBottomPanel } from '../components/dashboard/AnalyticsBottomPanel';
import { ProfileHistoryDrawer } from '../components/dashboard/ProfileHistoryDrawer';
import { useAnalysisStore, useAuthStore } from '../store';

export const Dashboard: React.FC = () => {
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  
  const latestResult = useAnalysisStore((s) => s.latestResult);
  const clearAuth = useAuthStore((s) => s.clearAuth);

  const dl = latestResult?.deep_learning || undefined;
  const expert = latestResult?.expert_system || undefined;
  const behavior = latestResult?.behavior || undefined;

  const handleLogout = () => {
    clearAuth();
    window.location.href = '/';
  };

  return (
    <div className="min-h-screen bg-dark-950 text-white flex flex-col p-4 lg:p-6" style={{ backgroundImage: 'var(--tw-bg-grid-pattern)' }}>
      {/* 3-Column Enterprise Dashboard Grid Layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-5 max-w-[1700px] mx-auto w-full">
        
        {/* ── COLUMN 1: LEFT SIDEBAR (Span 3) ─────────────────────────────────── */}
        <div className="lg:col-span-3 flex flex-col">
          <LeftSidebar 
            onProfileClick={() => setIsProfileOpen(true)}
            onLogout={handleLogout}
          />
        </div>

        {/* ── COLUMN 2: CENTER PANEL - MAIN COCKPIT (Span 6) ────────────────── */}
        <div className="lg:col-span-6 flex flex-col gap-5">
          {/* Top telemetry state bar */}
          <TopBar />

          {/* Primary 3D projection visualizer visual feed */}
          <PrimaryAIVisualizer />

          {/* Professional Tabbed bottom analytics panel */}
          <AnalyticsBottomPanel 
            dl={dl}
            expert={expert}
            behavior={behavior}
          />
        </div>

        {/* ── COLUMN 3: RIGHT PANEL - AI INSIGHT CARDS (Span 3) ─────────────── */}
        <div className="lg:col-span-3 flex flex-col">
          <RightPanel 
            dl={dl}
            expert={expert}
            behavior={behavior}
          />
        </div>
      </div>

      {/* Profile & History Drawer Overlay */}
      <ProfileHistoryDrawer
        isOpen={isProfileOpen}
        onClose={() => setIsProfileOpen(false)}
      />

      <footer className="mt-8 mb-2 text-center text-[10px] font-mono text-dark-500 flex items-center justify-center gap-1.5 border-t border-dark-800/40 pt-4">
        <span className="text-white font-semibold">© 2026</span>
        <span>Developed and Researched by</span>
        <a href="https://www.authbrain.io" target="_blank" rel="noopener noreferrer" className="text-white font-bold hover:underline">
          AuthBrain
        </a>
        <span>·</span>
        <a href="https://www.authbrain.io" target="_blank" rel="noopener noreferrer" className="text-dark-400 hover:text-white transition-colors hover:underline">
          www.authbrain.io
        </a>
      </footer>
    </div>
  );
};
