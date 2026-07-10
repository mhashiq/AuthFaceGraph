import React, { useState } from 'react';
import type { DLAnalysisResult, ExpertSystemResult, BehaviorResult, Landmark } from '../../types/analysis';
import { useAnalysisStore } from '../../store';
import { EmotionTimelineChart } from './EmotionTimelineChart';
import { EARChart, HeadPoseChart, FatigueChart } from '../charts/EARChart';
import { BarChart3, LineChart as LcIcon, Table, Compass, Activity, ActivitySquare, AlertCircle } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface AnalyticsBottomPanelProps {
  dl: DLAnalysisResult | undefined;
  expert: ExpertSystemResult | undefined;
  behavior: BehaviorResult | undefined;
}

export const AnalyticsBottomPanel: React.FC<AnalyticsBottomPanelProps> = ({ dl, expert, behavior }) => {
  const [activeTab, setActiveTab] = useState<string>('timeline');
  const emotionHistory = useAnalysisStore((s) => s.emotionHistory);
  const earHistory = useAnalysisStore((s) => s.earHistory);
  const headPoseHistory = useAnalysisStore((s) => s.headPoseHistory);
  const latestResult = useAnalysisStore((s) => s.latestResult);

  // Tab configurations
  const tabs = [
    { id: 'timeline', label: 'Emotion Timeline', icon: LcIcon },
    { id: 'probabilities', label: 'Emotion Probabilities', icon: BarChart3 },
    { id: 'actionUnits', label: 'Action Units (FACS)', icon: ActivitySquare },
    { id: 'nodeImportance', label: 'GNN Node Importance', icon: Table },
    { id: 'headPose', label: 'Head Pose (3D)', icon: Compass },
    { id: 'earMar', label: 'EAR / MAR Signal', icon: Activity },
    { id: 'ensemble', label: 'Model Ensemble', icon: Table },
  ];

  // Helper to format float values
  const pct = (val: number) => `${Math.round(val * 100)}%`;

  return (
    <div className="bg-dark-900 border border-dark-600/60 rounded-2xl p-5 shadow-2xl flex flex-col gap-4">
      {/* Tabs Header Navigation */}
      <div className="flex items-center border-b border-dark-700/60 overflow-x-auto scrollbar-none gap-2 pb-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 text-xs font-mono font-medium rounded-lg border transition-all whitespace-nowrap ${
                isActive
                  ? 'bg-violet-600/10 border-violet-500/40 text-violet-400 font-bold'
                  : 'bg-transparent border-transparent text-dark-400 hover:text-slate-200'
              }`}
            >
              <Icon size={13} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab Panels Content */}
      <div className="flex-1 min-h-[220px]">
        {/* TAB 1: EMOTION TIMELINE */}
        {activeTab === 'timeline' && (
          <div className="animate-in fade-in duration-200">
            {dl?.dl_enabled ? (
              <EmotionTimelineChart data={emotionHistory} />
            ) : (
              <p className="text-xs text-dark-400 font-mono py-8 text-center">
                Deep learning model is currently inactive. Start analysis to populate timeline.
              </p>
            )}
          </div>
        )}

        {/* TAB 2: EMOTION PROBABILITIES */}
        {activeTab === 'probabilities' && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 animate-in fade-in duration-200">
            {dl?.emotion_ensemble?.probabilities ? (
              Object.entries(dl.emotion_ensemble.probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([emotion, prob]) => {
                  const percentage = Math.round(prob * 100);
                  let barColor = "from-violet-600 to-indigo-400";
                  let textColor = "text-violet-400";
                  if (emotion === 'happy') { barColor = "from-emerald-600 to-teal-400"; textColor = "text-emerald-400"; }
                  else if (emotion === 'sad') { barColor = "from-blue-600 to-cyan-400"; textColor = "text-blue-400"; }
                  else if (emotion === 'anger') { barColor = "from-rose-600 to-red-400"; textColor = "text-rose-400"; }
                  else if (emotion === 'neutral') { barColor = "from-slate-600 to-slate-400"; textColor = "text-slate-400"; }

                  return (
                    <div key={emotion} className="bg-slate-950/40 border border-dark-600/40 p-4 rounded-xl flex flex-col justify-between gap-2 shadow-inner">
                      <div className="flex justify-between text-xs font-mono">
                        <span className={`capitalize font-bold ${textColor}`}>{emotion}</span>
                        <span className={`${textColor} font-extrabold`}>{percentage}%</span>
                      </div>
                      <div className="h-2 w-full bg-slate-950/60 rounded-full overflow-hidden border border-dark-600/10">
                        <div 
                          className={`h-full bg-gradient-to-r ${barColor} rounded-full`}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })
            ) : (
              <p className="col-span-full text-xs text-dark-400 font-mono py-8 text-center">
                Waiting for emotion classifier consensus results...
              </p>
            )}
          </div>
        )}

        {/* TAB 3: FACS ACTION UNITS */}
        {activeTab === 'actionUnits' && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 animate-in fade-in duration-200">
            {dl?.action_units && dl.action_units.length > 0 ? (
              dl.action_units.map((au) => {
                const intensityPercent = Math.round((au.intensity / 5.0) * 100);
                return (
                  <div key={au.au_id} className="bg-slate-950/30 border border-dark-600/30 p-3 rounded-xl font-mono text-[10px] space-y-1.5 shadow-sm">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="text-slate-200 font-bold">{au.au_id} ({au.name})</span>
                      <span className={au.present ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
                        {au.intensity.toFixed(1)} / 5.0
                      </span>
                    </div>
                    <div className="h-1 w-full bg-slate-950/60 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${au.present ? 'bg-violet-500' : 'bg-slate-700'}`}
                        style={{ width: `${intensityPercent}%` }}
                      />
                    </div>
                  </div>
                );
              })
            ) : (
              <p className="col-span-full text-xs text-dark-400 font-mono py-8 text-center">
                Action Unit muscles estimator not computed.
              </p>
            )}
          </div>
        )}

        {/* TAB 4: GNN NODE IMPORTANCE */}
        {activeTab === 'nodeImportance' && (
          <div className="overflow-x-auto scrollbar-thin animate-in fade-in duration-200">
            {dl?.top_important_landmarks && dl.top_important_landmarks.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3 max-h-[220px] overflow-y-auto pr-1">
                {dl.top_important_landmarks.slice(0, 15).map((nodeId, index) => {
                  const val = dl.gnn_prediction?.node_importance?.[nodeId] ?? 0;
                  const lm = dl.landmarks[nodeId] || { x: 0, y: 0, z: 0 };
                  
                  // Anatomical region mapping helper
                  const getRegion = (id: number) => {
                    if (id < 33 || (id > 132 && id < 162)) return 'Left Eye';
                    if (id > 361 && id < 398) return 'Right Eye';
                    if (id > 50 && id < 125) return 'Eyebrow';
                    if (id > 170 && id < 290) return 'Lips';
                    return 'Face Outline';
                  };

                  return (
                    <div 
                      key={nodeId} 
                      className="bg-slate-950/40 border border-dark-600/40 hover:border-violet-500/30 p-3 rounded-xl font-mono text-[9px] flex flex-col gap-1.5 shadow-sm transition-all"
                    >
                      <div className="flex justify-between border-b border-dark-600/20 pb-1">
                        <span className="text-violet-400 font-bold">Node #{nodeId}</span>
                        <span className="text-slate-400">Rank #{index + 1}</span>
                      </div>
                      <div>Region: <span className="text-slate-200 font-bold">{getRegion(nodeId)}</span></div>
                      <div className="flex justify-between">
                        <span>Importance:</span>
                        <span className="text-violet-400 font-bold">{pct(val)}</span>
                      </div>
                      <div className="text-slate-500">
                        ({lm.x.toFixed(2)}, {lm.y.toFixed(2)}, {lm.z?.toFixed(2) ?? 0})
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-dark-400 font-mono py-8 text-center">
                Awaiting active GNN prediction attributions to index importance.
              </p>
            )}
          </div>
        )}

        {/* TAB 5: HEAD POSE */}
        {activeTab === 'headPose' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 animate-in fade-in duration-200">
            {/* Visualizer component placeholder */}
            <div className="bg-slate-950/40 border border-dark-600/40 rounded-xl p-4 flex flex-col justify-center">
              <span className="text-[10px] text-dark-300 font-mono uppercase block mb-3">
                Orientation Coordinates
              </span>
              <div className="grid grid-cols-3 gap-3 text-center font-mono">
                <div className="bg-slate-950/60 p-2.5 rounded-lg border border-dark-600/20">
                  <span className="text-[9px] text-dark-500 block">PITCH</span>
                  <span className="text-xs text-rose-400 font-bold">
                    {latestResult?.head_pose ? `${latestResult.head_pose.pitch.toFixed(1)}°` : '—'}
                  </span>
                </div>
                <div className="bg-slate-950/60 p-2.5 rounded-lg border border-dark-600/20">
                  <span className="text-[9px] text-dark-500 block">YAW</span>
                  <span className="text-xs text-emerald-400 font-bold">
                    {latestResult?.head_pose ? `${latestResult.head_pose.yaw.toFixed(1)}°` : '—'}
                  </span>
                </div>
                <div className="bg-slate-950/60 p-2.5 rounded-lg border border-dark-600/20">
                  <span className="text-[9px] text-dark-500 block">ROLL</span>
                  <span className="text-xs text-violet-400 font-bold">
                    {latestResult?.head_pose ? `${latestResult.head_pose.roll.toFixed(1)}°` : '—'}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex-1 min-h-[140px]">
              <HeadPoseChart />
            </div>
          </div>
        )}

        {/* TAB 6: EAR / MAR SIGNAL */}
        {activeTab === 'earMar' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-in fade-in duration-200">
            <div className="h-[200px]">
              <EARChart />
            </div>
            <div className="bg-slate-950/40 border border-dark-600/40 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <span className="text-[10px] text-dark-300 font-mono uppercase block mb-2">
                  Signal Threshold Metrics
                </span>
                <div className="space-y-2 font-mono text-[10px]">
                  <div className="flex justify-between border-b border-dark-600/20 pb-1">
                    <span className="text-slate-400">Eye Aspect Ratio (EAR):</span>
                    <span className="text-emerald-400 font-bold">
                      {latestResult?.eyes ? latestResult.eyes.average_ear.toFixed(3) : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-dark-600/20 pb-1">
                    <span className="text-slate-400">Mouth Aspect Ratio (MAR):</span>
                    <span className="text-violet-400 font-bold">
                      {latestResult?.mouth ? latestResult.mouth.mar.toFixed(3) : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Lip Smile Intensity:</span>
                    <span className="text-indigo-400 font-bold">
                      {latestResult?.mouth ? `${Math.round(latestResult.mouth.smile_intensity * 100)}%` : '—'}
                    </span>
                  </div>
                </div>
              </div>
              {latestResult?.eyes && latestResult.eyes.eye_closure_duration_ms > 0 && (
                <div className="bg-rose-500/10 border border-rose-500/20 p-2.5 rounded-lg flex items-center gap-2 text-[10px] font-mono text-rose-400">
                  <AlertCircle size={12} />
                  <span>Drowsy closure detected: {latestResult.eyes.eye_closure_duration_ms.toFixed(0)} ms</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 7: MODEL ENSEMBLE */}
        {activeTab === 'ensemble' && (
          <div className="overflow-x-auto scrollbar-thin animate-in fade-in duration-200">
            {dl?.emotion_ensemble?.model_predictions ? (
              <table className="w-full text-left font-mono text-[10px] text-slate-300 border-collapse">
                <thead>
                  <tr className="border-b border-dark-600/60 text-dark-400">
                    <th className="py-2 pr-4">MODEL IDENTIFIER</th>
                    <th className="py-2 pr-4">PREDICTION</th>
                    <th className="py-2 pr-4">CONFIDENCE</th>
                    <th className="py-2 pr-4">LATENCY</th>
                    <th className="py-2 pr-4">UNCERTAINTY FLAG</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-700/30">
                  {/* Ensemble entry */}
                  <tr className="text-violet-400 font-bold">
                    <td className="py-2.5 pr-4 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-violet-400" />
                      Ensemble Consensus
                    </td>
                    <td className="py-2.5 pr-4 capitalize">{latestResult?.expert_system?.attention_state ?? 'focused'}</td>
                    <td className="py-2.5 pr-4">{pct(latestResult?.model_confidence ?? 0)}</td>
                    <td className="py-2.5 pr-4">{latestResult?.inference_time_ms.toFixed(1)} ms</td>
                    <td className="py-2.5 pr-4 text-emerald-400">Consensus Achieved</td>
                  </tr>

                  {/* Individual models predictions */}
                  {dl.emotion_ensemble.model_predictions.map((pred) => (
                    <tr key={pred.model_id}>
                      <td className="py-2.5 pr-4 pl-3 text-slate-400">{pred.model_id}</td>
                      <td className="py-2.5 pr-4 capitalize text-slate-200">{pred.emotion}</td>
                      <td className="py-2.5 pr-4 text-slate-300">{pct(pred.confidence)}</td>
                      <td className="py-2.5 pr-4 text-slate-400">{pred.latency_ms.toFixed(1)} ms</td>
                      <td className="py-2.5 pr-4 text-slate-400">—</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-xs text-dark-400 font-mono py-8 text-center">
                No ensemble models loaded in this session.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
