/**
 * AuthFaceGraph — Analytics Section
 * Deep metrics, emotion trends, performance graphs
 */

import React, { useMemo } from 'react';
import { useAnalysisStore } from '../../store';
import { BarChart3, TrendingUp, Brain, Activity, Zap, Target } from 'lucide-react';
import { GlassCard, SectionHeader, LoadingSkeleton } from '../../components/ui';
import {
  AreaChart, Area, BarChart, Bar, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';

const EMOTION_COLORS: Record<string,string> = {
  Happy:'#10b981', Sad:'#3b82f6', Angry:'#ef4444', Fear:'#f59e0b',
  Surprise:'#8b5cf6', Disgust:'#ec4899', Neutral:'#6b7280',
};

export const AnalyticsDashboard: React.FC = () => {
  const history = useAnalysisStore(s => s.history);
  const latest  = useAnalysisStore(s => s.latestResult);

  const trendData = useMemo(() => history.slice(-30).map((r, i) => ({
    i,
    emotion: r.deep_learning?.emotion_ensemble?.confidence ?? 0,
    fps: r.fps ?? 0,
    latency: r.inference_time_ms ?? 0,
  })), [history]);

  const emotionDist = useMemo(() => {
    const counts: Record<string,number> = {};
    history.forEach(r => {
      const e = r.deep_learning?.emotion_ensemble?.final_emotion ?? 'Neutral';
      counts[e] = (counts[e] || 0) + 1;
    });
    return Object.entries(counts).map(([emotion, count]) => ({ emotion, count }));
  }, [history]);

  const radarData = useMemo(() => {
    const probs = latest?.deep_learning?.emotion_ensemble?.probabilities || {};
    return Object.entries(probs).map(([emotion, value]) => ({
      emotion, value: Math.round((value as number) * 100),
    }));
  }, [latest]);

  const dl = latest?.deep_learning;

  return (
    <div className="space-y-5 stagger-children">
      <SectionHeader
        title="Analytics & Insights"
        subtitle="Deep learning model performance and emotion analytics"
        icon={<BarChart3 size={16} />}
      />

      {/* ── KPI Row ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Sessions Analyzed', value: history.length, color: '#00d4ff', icon: <Activity size={18}/> },
          { label: 'Avg Confidence', value: `${Math.round((history.reduce((a,r) => a + (r.deep_learning?.emotion_ensemble?.confidence ?? 0), 0) / Math.max(history.length,1)) * 100)}%`, color: '#8b5cf6', icon: <Target size={18}/> },
          { label: 'Current FPS', value: `${(latest?.fps ?? 0).toFixed(0)}`, color: '#10b981', icon: <Zap size={18}/> },
          { label: 'Avg Latency', value: `${(history.reduce((a,r) => a + (r.inference_time_ms ?? 0), 0) / Math.max(history.length,1)).toFixed(0)}ms`, color: '#f59e0b', icon: <TrendingUp size={18}/> },
        ].map((kpi, i) => (
          <GlassCard key={i} className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: `${kpi.color}18`, border: `1px solid ${kpi.color}30` }}>
                <span style={{ color: kpi.color }}>{kpi.icon}</span>
              </div>
              <div>
                <div className="font-mono text-[10px] text-slate-400 uppercase tracking-wider">{kpi.label}</div>
                <div className="font-bold text-xl mt-0.5" style={{ color: kpi.color }}>{kpi.value}</div>
              </div>
            </div>
          </GlassCard>
        ))}
      </div>

      {/* ── Charts Row ──────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Confidence Trend */}
        <GlassCard className="p-5">
          <div className="font-mono text-[10px] uppercase tracking-widest text-cyan-400/70 mb-4">
            Emotion Confidence Trend
          </div>
          {trendData.length > 2 ? (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="i" hide />
                <YAxis domain={[0,1]} hide />
                <Tooltip
                  contentStyle={{ background: '#0a0f1e', border: '1px solid rgba(0,212,255,0.2)', borderRadius: 10, fontSize: 11 }}
                  formatter={(v: number) => [`${Math.round(v*100)}%`, 'Confidence']}
                />
                <Area type="monotone" dataKey="emotion" stroke="#00d4ff" strokeWidth={2}
                  fill="url(#confGrad)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[180px] flex items-center justify-center">
              <LoadingSkeleton className="w-full h-full" />
            </div>
          )}
        </GlassCard>

        {/* Emotion Distribution */}
        <GlassCard className="p-5">
          <div className="font-mono text-[10px] uppercase tracking-widest text-violet-400/70 mb-4">
            Emotion Distribution
          </div>
          {emotionDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={emotionDist} barSize={16}>
                <XAxis dataKey="emotion" tick={{ fontSize: 9, fill: '#5d7399' }} />
                <YAxis hide />
                <Tooltip
                  contentStyle={{ background: '#0a0f1e', border: '1px solid rgba(139,92,246,0.2)', borderRadius: 10, fontSize: 11 }}
                />
                <Bar dataKey="count" fill="#7c3aed" radius={[4, 4, 0, 0]}
                  style={{ filter: 'drop-shadow(0 0 6px rgba(139,92,246,0.4))' }} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[180px] flex items-center justify-center text-slate-500 text-xs font-mono">
              No session data yet — start analysis
            </div>
          )}
        </GlassCard>

        {/* FPS & Latency */}
        <GlassCard className="p-5">
          <div className="font-mono text-[10px] uppercase tracking-widest text-green-400/70 mb-4">
            Performance — FPS vs Latency
          </div>
          {trendData.length > 2 ? (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="fpsGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="latGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="i" hide />
                <YAxis hide />
                <Tooltip contentStyle={{ background: '#0a0f1e', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 10, fontSize: 11 }} />
                <Area type="monotone" dataKey="fps" stroke="#10b981" strokeWidth={2} fill="url(#fpsGrad)" dot={false} name="FPS" />
                <Area type="monotone" dataKey="latency" stroke="#f59e0b" strokeWidth={2} fill="url(#latGrad)" dot={false} name="Latency (ms)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : <LoadingSkeleton className="h-[180px]" />}
        </GlassCard>

        {/* Emotion Radar */}
        <GlassCard className="p-5">
          <div className="font-mono text-[10px] uppercase tracking-widest text-blue-400/70 mb-4">
            Emotion Probability Radar
          </div>
          {radarData.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(79,70,229,0.2)" />
                <PolarAngleAxis dataKey="emotion" tick={{ fontSize: 9, fill: '#5d7399' }} />
                <Radar name="Emotion" dataKey="value" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.25}
                  dot={{ fill: '#8b5cf6', r: 2 }} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[180px] flex items-center justify-center text-slate-500 text-xs font-mono">
              Awaiting analysis data
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  );
};
