/**
 * AuthFaceGraph — Activity & Monitoring Section
 * Real-time system logs, alert feed, and health metrics
 */

import React, { useRef, useEffect } from 'react';
import { Activity, AlertTriangle, CheckCircle, Info, Cpu, MemoryStick, Clock } from 'lucide-react';
import { GlassCard, SectionHeader, StatusDot } from '../../components/ui';
import { useAnalysisStore } from '../../store';
import clsx from 'clsx';

export const ActivityMonitoring: React.FC = () => {
  const logs         = useAnalysisStore(s => s.logs);
  const activeAlerts = useAnalysisStore(s => s.activeAlerts);
  const latestResult = useAnalysisStore(s => s.latestResult);
  const wsState      = useAnalysisStore(s => s.wsState);
  const logEndRef    = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const fps      = latestResult?.fps ?? 0;
  const latency  = latestResult?.inference_time_ms ?? 0;

  const logLevelConfig = {
    info:    { icon: <Info size={11} />,          color: '#00d4ff', bg: 'rgba(0,212,255,0.08)' },
    warning: { icon: <AlertTriangle size={11} />, color: '#f59e0b', bg: 'rgba(245,158,11,0.08)' },
    error:   { icon: <AlertTriangle size={11} />, color: '#ef4444', bg: 'rgba(239,68,68,0.08)' },
    success: { icon: <CheckCircle size={11} />,   color: '#10b981', bg: 'rgba(16,185,129,0.08)' },
  };

  return (
    <div className="space-y-5 stagger-children">
      <SectionHeader
        title="Activity & Monitoring"
        subtitle="Real-time system logs, alerts, and health metrics"
        icon={<Activity size={16} />}
      />

      {/* Health Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          {
            label: 'WebSocket', icon: <Activity size={16}/>, value: wsState,
            status: wsState === 'connected' ? 'online' : wsState === 'connecting' ? 'warning' : 'offline' as any,
          },
          {
            label: 'FPS Rate', icon: <Cpu size={16}/>,
            value: `${fps.toFixed(0)} FPS`,
            status: fps > 20 ? 'online' : fps > 10 ? 'warning' : 'offline' as any,
          },
          {
            label: 'Inference', icon: <Clock size={16}/>,
            value: `${latency.toFixed(0)}ms`,
            status: latency < 100 ? 'online' : latency < 300 ? 'warning' : 'error' as any,
          },
          {
            label: 'Alerts', icon: <AlertTriangle size={16}/>,
            value: `${activeAlerts.length} active`,
            status: activeAlerts.length === 0 ? 'online' : 'warning' as any,
          },
        ].map((m, i) => (
          <GlassCard key={i} className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-slate-400">{m.icon}</span>
              <span className="font-mono text-[9px] text-slate-400 uppercase tracking-wider">{m.label}</span>
            </div>
            <div className="text-sm font-bold text-white mb-1">{m.value}</div>
            <StatusDot status={m.status} />
          </GlassCard>
        ))}
      </div>

      {/* Active Alerts */}
      {activeAlerts.length > 0 && (
        <GlassCard className="p-5">
          <div className="font-mono text-[10px] uppercase tracking-widest text-red-400/70 mb-4">
            ⚠ Active Alerts ({activeAlerts.length})
          </div>
          <div className="space-y-2">
            {activeAlerts.map((alert, i) => (
              <div key={i} className="flex items-center gap-3 p-3 rounded-xl"
                style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)' }}>
                <AlertTriangle size={13} className="text-red-400 flex-shrink-0" />
                <span className="text-xs text-red-300">{alert}</span>
              </div>
            ))}
          </div>
        </GlassCard>
      )}

      {/* System Log Feed */}
      <GlassCard className="p-5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-slate-400 mb-4">
          System Log — Live Feed
        </div>
        <div className="space-y-1.5 max-h-96 overflow-y-auto pr-1">
          {logs.length === 0 ? (
            <div className="text-center py-8 text-slate-500 text-xs font-mono">
              No logs yet — system idle
            </div>
          ) : (
            logs.slice(-80).map((log, i) => {
              const cfg = logLevelConfig[log.level as keyof typeof logLevelConfig] || logLevelConfig.info;
              return (
                <div
                  key={i}
                  className="flex items-start gap-2.5 p-2 rounded-lg text-[10px] font-mono transition-all"
                  style={{ background: cfg.bg }}
                >
                  <span style={{ color: cfg.color, flexShrink: 0, marginTop: 1 }}>{cfg.icon}</span>
                  <span className="text-slate-400 flex-shrink-0 tabular-nums">[{log.timestamp}]</span>
                  <span className="text-slate-500 flex-shrink-0 uppercase">{log.source}</span>
                  <span style={{ color: cfg.color }}>{log.message}</span>
                </div>
              );
            })
          )}
          <div ref={logEndRef} />
        </div>
      </GlassCard>
    </div>
  );
};
