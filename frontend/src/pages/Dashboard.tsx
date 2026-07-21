/**
 * AuthFaceGraph AI
 * Main Dashboard Shell — 5-Section Navigation
 *
 * Premium futuristic AI operating system interface with
 * section-based navigation, smooth transitions, and zero blank screens.
 */

import React, { useState, useCallback } from 'react';
import { LeftSidebar } from '../components/dashboard/LeftSidebar';
import { TopBar }      from '../components/dashboard/TopBar';
import { PrimaryAIVisualizer } from '../components/webcam/PrimaryAIVisualizer';
import { RightPanel }          from '../components/dashboard/RightPanel';
import { AnalyticsBottomPanel } from '../components/dashboard/AnalyticsBottomPanel';
import { ProfileHistoryDrawer } from '../components/dashboard/ProfileHistoryDrawer';
import { PageTransition }       from '../components/ui';
import { AnalyticsDashboard }   from './sections/AnalyticsDashboard';
import { UserManagement }       from './sections/UserManagement';
import { ActivityMonitoring }   from './sections/ActivityMonitoring';
import { SettingsPanel }        from './sections/SettingsPanel';
import { useAnalysisStore, useAuthStore } from '../store';

type Section = 'dashboard' | 'analytics' | 'users' | 'activity' | 'settings';

export const Dashboard: React.FC = () => {
  const [activeSection, setActiveSection] = useState<Section>('dashboard');
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isExiting, setIsExiting] = useState(false);

  const latestResult = useAnalysisStore(s => s.latestResult);
  const clearAuth    = useAuthStore(s => s.clearAuth);

  const dl       = latestResult?.deep_learning || undefined;
  const expert   = latestResult?.expert_system || undefined;
  const behavior = latestResult?.behavior || undefined;

  const handleLogout = useCallback(() => {
    setIsExiting(true);
    setTimeout(() => {
      clearAuth();
      window.location.href = '/';
    }, 350);
  }, [clearAuth]);

  const handleSectionChange = useCallback((section: Section) => {
    setActiveSection(section);
  }, []);

  return (
    <div
      className="min-h-screen flex gpu-accelerated"
      style={{
        background: 'linear-gradient(135deg, #010409 0%, #030712 50%, #070d1a 100%)',
        opacity: isExiting ? 0 : 1,
        transition: 'opacity 0.35s ease',
      }}
    >
      {/* Ambient background effects */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {/* Grid */}
        <div className="absolute inset-0 bg-grid opacity-40" />
        {/* Ambient glow top-left */}
        <div className="absolute top-0 left-1/4 w-[600px] h-[400px] opacity-[0.04]"
          style={{ background: 'radial-gradient(ellipse, #7c3aed, transparent 70%)', filter: 'blur(40px)' }} />
        {/* Ambient glow bottom-right */}
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[300px] opacity-[0.04]"
          style={{ background: 'radial-gradient(ellipse, #00d4ff, transparent 70%)', filter: 'blur(40px)' }} />
      </div>

      {/* ── SIDEBAR ─────────────────────────────────────────────── */}
      <div className="relative z-10 w-64 flex-shrink-0 p-4 flex flex-col">
        <LeftSidebar
          activeSection={activeSection}
          onSectionChange={handleSectionChange}
          onProfileClick={() => setIsProfileOpen(true)}
          onLogout={handleLogout}
        />
      </div>

      {/* ── MAIN CONTENT AREA ──────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 p-4 pl-2 gap-4 relative z-10 overflow-hidden">
        {/* TopBar — always visible */}
        <TopBar activeSection={activeSection} />

        {/* Content scroll container */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden pr-1 -mr-1">

          {/* ── DASHBOARD SECTION ─────────────────────────────── */}
          {activeSection === 'dashboard' && (
            <PageTransition id="dashboard">
              <div className="flex gap-4 h-full">
                {/* Center: AI Visualizer + Analytics */}
                <div className="flex-1 flex flex-col gap-4 min-w-0">
                  <PrimaryAIVisualizer />
                  <AnalyticsBottomPanel dl={dl} expert={expert} behavior={behavior} />
                </div>

                {/* Right Panel */}
                <div className="w-[340px] flex-shrink-0">
                  <RightPanel dl={dl} expert={expert} behavior={behavior} />
                </div>
              </div>
            </PageTransition>
          )}

          {/* ── ANALYTICS SECTION ─────────────────────────────── */}
          {activeSection === 'analytics' && (
            <PageTransition id="analytics">
              <AnalyticsDashboard />
            </PageTransition>
          )}

          {/* ── USER MANAGEMENT SECTION ───────────────────────── */}
          {activeSection === 'users' && (
            <PageTransition id="users">
              <UserManagement />
            </PageTransition>
          )}

          {/* ── ACTIVITY & MONITORING SECTION ─────────────────── */}
          {activeSection === 'activity' && (
            <PageTransition id="activity">
              <ActivityMonitoring />
            </PageTransition>
          )}

          {/* ── SETTINGS SECTION ──────────────────────────────── */}
          {activeSection === 'settings' && (
            <PageTransition id="settings">
              <SettingsPanel />
            </PageTransition>
          )}
        </div>
      </div>

      {/* Profile History Drawer */}
      <ProfileHistoryDrawer
        isOpen={isProfileOpen}
        onClose={() => setIsProfileOpen(false)}
      />
    </div>
  );
};
