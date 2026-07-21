/**
 * AuthFaceGraph — Settings Panel Section
 * System configuration, model params, API settings
 */

import React, { useState } from 'react';
import { Settings, Sliders, Server, Shield, Bell, Monitor, Save } from 'lucide-react';
import { GlassCard, SectionHeader, NeonButton } from '../../components/ui';

interface ToggleProps {
  label: string;
  description?: string;
  value: boolean;
  onChange: (v: boolean) => void;
  color?: string;
}

const Toggle: React.FC<ToggleProps> = ({ label, description, value, onChange, color = '#4f46e5' }) => (
  <div className="flex items-center justify-between py-3 border-b border-indigo-500/08 last:border-0">
    <div>
      <div className="text-sm font-medium text-slate-200">{label}</div>
      {description && <div className="font-mono text-[10px] text-slate-500 mt-0.5">{description}</div>}
    </div>
    <button
      onClick={() => onChange(!value)}
      className="relative w-11 h-6 rounded-full transition-all duration-300 flex-shrink-0"
      style={{
        background: value ? `linear-gradient(135deg, ${color}, #7c3aed)` : 'rgba(30,40,70,0.8)',
        border: `1px solid ${value ? color + '60' : 'rgba(79,70,229,0.2)'}`,
        boxShadow: value ? `0 0 12px ${color}40` : 'none',
      }}
    >
      <div className="absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-all duration-300 shadow-md"
        style={{ transform: value ? 'translateX(20px)' : 'translateX(0)' }} />
    </button>
  </div>
);

export const SettingsPanel: React.FC = () => {
  // Analysis settings
  const [detectEmotions, setDetectEmotions]     = useState(true);
  const [detectLandmarks, setDetectLandmarks]   = useState(true);
  const [detectAttention, setDetectAttention]   = useState(true);
  const [expertSystem, setExpertSystem]         = useState(true);
  const [xaiEnabled, setXaiEnabled]             = useState(true);
  const [showGNNGraph, setShowGNNGraph]          = useState(true);

  // Performance
  const [targetFps, setTargetFps]               = useState(30);
  const [useOnnx, setUseOnnx]                   = useState(true);
  const [enableGpu, setEnableGpu]               = useState(false);
  const [streamCompression, setStreamCompression] = useState(true);

  // Privacy
  const [autoDelete, setAutoDelete]             = useState(true);
  const [encrypted, setEncrypted]               = useState(true);
  const [auditLogs, setAuditLogs]               = useState(true);

  // Notifications
  const [alertsEnabled, setAlertsEnabled]       = useState(true);
  const [soundAlerts, setSoundAlerts]           = useState(false);
  const [criticalOnly, setCriticalOnly]         = useState(false);

  // API
  const [apiBase, setApiBase] = useState(import.meta.env.VITE_API_URL || 'http://localhost:8000');
  const [wsEndpoint, setWsEndpoint] = useState(import.meta.env.VITE_WS_URL || 'ws://localhost:8000');

  const sections = [
    {
      title: 'Analysis Engine',
      icon: <Sliders size={15} />,
      content: (
        <div>
          <Toggle label="Emotion Detection"   description="Real-time facial emotion classification" value={detectEmotions}   onChange={setDetectEmotions}  color="#00d4ff" />
          <Toggle label="Landmark Tracking"   description="106-point facial landmark detection"      value={detectLandmarks}  onChange={setDetectLandmarks} color="#00d4ff" />
          <Toggle label="Attention Analysis"  description="Gaze direction & focus state tracking"    value={detectAttention}  onChange={setDetectAttention} color="#8b5cf6" />
          <Toggle label="Expert System"       description="Rule-based cognitive state engine"         value={expertSystem}     onChange={setExpertSystem}    color="#8b5cf6" />
          <Toggle label="XAI Explanations"    description="Explainable AI justification overlays"    value={xaiEnabled}       onChange={setXaiEnabled}      color="#f59e0b" />
          <Toggle label="GNN Graph View"      description="Graph neural network visualization"       value={showGNNGraph}     onChange={setShowGNNGraph}    color="#3b82f6" />
        </div>
      ),
    },
    {
      title: 'Performance',
      icon: <Monitor size={15} />,
      content: (
        <div>
          <div className="py-3 border-b border-indigo-500/08">
            <div className="flex items-center justify-between mb-2">
              <div>
                <div className="text-sm font-medium text-slate-200">Target Frame Rate</div>
                <div className="font-mono text-[10px] text-slate-500">Webcam capture FPS</div>
              </div>
              <span className="font-mono text-sm font-bold text-cyan-400">{targetFps} FPS</span>
            </div>
            <input
              type="range" min={5} max={60} step={5} value={targetFps}
              onChange={e => setTargetFps(Number(e.target.value))}
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
              style={{ background: `linear-gradient(to right, #00d4ff ${((targetFps - 5)/55)*100}%, rgba(79,70,229,0.2) ${((targetFps - 5)/55)*100}%)` }}
            />
          </div>
          <Toggle label="ONNX Runtime"     description="Optimized neural inference engine" value={useOnnx}            onChange={setUseOnnx}            color="#10b981" />
          <Toggle label="GPU Acceleration" description="CUDA/Metal hardware acceleration"   value={enableGpu}          onChange={setEnableGpu}          color="#10b981" />
          <Toggle label="Stream Compress"  description="Optimize WebSocket payloads"        value={streamCompression}  onChange={setStreamCompression}  color="#3b82f6" />
        </div>
      ),
    },
    {
      title: 'Privacy & Security',
      icon: <Shield size={15} />,
      content: (
        <div>
          <Toggle label="Auto-delete Session Data" description="Purge biometric data on logout"  value={autoDelete}  onChange={setAutoDelete}  color="#ef4444" />
          <Toggle label="Transport Encryption"     description="AES-256 / TLS 1.3 encryption"    value={encrypted}   onChange={setEncrypted}   color="#10b981" />
          <Toggle label="Audit Log Retention"      description="Keep security audit trails"       value={auditLogs}   onChange={setAuditLogs}   color="#f59e0b" />
        </div>
      ),
    },
    {
      title: 'Notifications & Alerts',
      icon: <Bell size={15} />,
      content: (
        <div>
          <Toggle label="Enable Alerts"    description="System alert notifications"             value={alertsEnabled}  onChange={setAlertsEnabled}  color="#8b5cf6" />
          <Toggle label="Sound Alerts"     description="Audible alert for critical events"      value={soundAlerts}    onChange={setSoundAlerts}    color="#f59e0b" />
          <Toggle label="Critical Only"    description="Only show critical severity alerts"     value={criticalOnly}   onChange={setCriticalOnly}   color="#ef4444" />
        </div>
      ),
    },
    {
      title: 'API & Connectivity',
      icon: <Server size={15} />,
      content: (
        <div className="space-y-4 pt-2">
          <div>
            <label className="font-mono text-[10px] uppercase tracking-widest text-slate-400 block mb-1.5">
              Backend API URL
            </label>
            <input
              type="text"
              value={apiBase}
              onChange={e => setApiBase(e.target.value)}
              className="neon-input w-full px-4 py-2.5 rounded-xl text-sm font-mono"
            />
          </div>
          <div>
            <label className="font-mono text-[10px] uppercase tracking-widest text-slate-400 block mb-1.5">
              WebSocket Endpoint
            </label>
            <input
              type="text"
              value={wsEndpoint}
              onChange={e => setWsEndpoint(e.target.value)}
              className="neon-input w-full px-4 py-2.5 rounded-xl text-sm font-mono"
            />
          </div>
          <NeonButton variant="secondary" size="sm">
            Test Connection
          </NeonButton>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-5 stagger-children">
      <SectionHeader
        title="System Settings"
        subtitle="Configure analysis engine, performance, and security"
        icon={<Settings size={16} />}
        actions={
          <NeonButton size="sm" variant="primary">
            <Save size={13} /> Save Changes
          </NeonButton>
        }
      />

      {sections.map((section, i) => (
        <GlassCard key={i} className="p-5">
          <div className="flex items-center gap-2.5 mb-4 pb-3 border-b border-indigo-500/10">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(79,70,229,0.15)', border: '1px solid rgba(79,70,229,0.3)' }}>
              <span className="text-violet-400">{section.icon}</span>
            </div>
            <span className="font-semibold text-sm text-slate-200">{section.title}</span>
          </div>
          {section.content}
        </GlassCard>
      ))}
    </div>
  );
};
