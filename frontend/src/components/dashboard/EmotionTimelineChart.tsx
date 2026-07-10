import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

export interface EmotionTimelineDataPoint {
  time: number;
  happy: number;
  sad: number;
  neutral: number;
  surprise: number;
  anger: number;
}

interface EmotionTimelineChartProps {
  data: EmotionTimelineDataPoint[];
}

export const EmotionTimelineChart: React.FC<EmotionTimelineChartProps> = ({ data }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl transition-all duration-300 hover:border-violet-500/50">
      <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-violet-400 animate-pulse" />
        Emotion Evolution Timeline
      </h3>
      <div className="h-64 w-full">
        {data.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500 text-sm">
            Waiting for timeline frames...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={data}
              margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
            >
              <defs>
                <linearGradient id="colorHappy" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="colorNeutral" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="colorSad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="time"
                stroke="#475569"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
              />
              <YAxis
                domain={[0, 1]}
                stroke="#475569"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '12px',
                }}
                labelStyle={{ color: '#94a3b8', fontSize: 11 }}
                itemStyle={{ fontSize: 12 }}
              />
              <Area
                type="monotone"
                dataKey="neutral"
                stroke="#6366f1"
                fillOpacity={1}
                fill="url(#colorNeutral)"
                name="Neutral"
              />
              <Area
                type="monotone"
                dataKey="happy"
                stroke="#10b981"
                fillOpacity={1}
                fill="url(#colorHappy)"
                name="Happy"
              />
              <Area
                type="monotone"
                dataKey="sad"
                stroke="#3b82f6"
                fillOpacity={1}
                fill="url(#colorSad)"
                name="Sad"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};
