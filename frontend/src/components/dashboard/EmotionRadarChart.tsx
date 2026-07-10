import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts';

interface EmotionRadarChartProps {
  probabilities: Record<string, number>;
}

export const EmotionRadarChart: React.FC<EmotionRadarChartProps> = ({ probabilities }) => {
  // Format data for Recharts RadarChart
  const data = Object.entries(probabilities).map(([emotion, value]) => ({
    subject: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    value: Math.round(value * 100),
  }));

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl transition-all duration-300 hover:border-violet-500/50">
      <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-violet-500 animate-pulse" />
        Emotion Probabilities
      </h3>
      <div className="h-64 w-full flex items-center justify-center">
        {data.length === 0 ? (
          <p className="text-slate-500 text-sm">No emotion data available</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis
                dataKey="subject"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
              />
              <PolarRadiusAxis
                angle={30}
                domain={[0, 100]}
                tick={{ fill: '#64748b', fontSize: 10 }}
              />
              <Radar
                name="Confidence"
                dataKey="value"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.25}
              />
            </RadarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};
