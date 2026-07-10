/**
 * AuthBrain AI Face Analysis Engine
 * CameraFeed Component
 *
 * Displays the live webcam feed with annotated overlay from backend.
 * Shows the face mesh JPEG returned by the backend over the raw video.
 */

import React, { useRef, useEffect } from 'react';
import { Camera, CameraOff, Wifi, WifiOff, AlertTriangle } from 'lucide-react';
import { useAnalysisStore } from '../../store';
import { useFaceAnalysis } from '../../hooks/useFaceAnalysis';
import clsx from 'clsx';

interface CameraFeedProps {
  className?: string;
}

const ConnectionBadge: React.FC = () => {
  const wsState = useAnalysisStore(s => s.wsState);
  const configs = {
    connected:    { label: 'LIVE',         icon: Wifi,    cls: 'bg-brand-500/20 text-brand-500 border-brand-500/40' },
    connecting:   { label: 'CONNECTING',   icon: Wifi,    cls: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40 animate-pulse' },
    disconnected: { label: 'DISCONNECTED', icon: WifiOff, cls: 'bg-gray-500/20 text-gray-400 border-gray-500/40' },
    error:        { label: 'ERROR',        icon: WifiOff, cls: 'bg-risk-critical/20 text-risk-critical border-risk-critical/40' },
  };
  const { label, icon: Icon, cls } = configs[wsState];
  return (
    <span className={clsx('inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-mono font-semibold border', cls)}>
      <Icon size={10} />
      {label}
    </span>
  );
};

export const CameraFeed: React.FC<CameraFeedProps> = ({ className }) => {
  const { videoRef, isRunning, cameraError, start, stop } = useFaceAnalysis();
  const annotatedFrameUrl = useAnalysisStore(s => s.annotatedFrameUrl);
  const latestResult      = useAnalysisStore(s => s.latestResult);
  const activeAlerts      = useAnalysisStore(s => s.activeAlerts);

  return (
    <div className={clsx('relative flex flex-col', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Camera size={16} className="text-brand-500" />
          <span className="text-sm font-semibold text-white font-mono uppercase tracking-wider">
            Live Feed
          </span>
        </div>
        <div className="flex items-center gap-3">
          {latestResult && (
            <span className="text-xs font-mono text-dark-200">
              {latestResult.fps.toFixed(1)} FPS
            </span>
          )}
          <ConnectionBadge />
        </div>
      </div>

      {/* Video Container */}
      <div className="relative rounded-xl overflow-hidden bg-dark-900 border border-dark-600 flex-1 min-h-0 aspect-video">
        {/* Raw video (always present, hidden when showing annotated frame) */}
        <video
          ref={videoRef}
          className={clsx(
            'absolute inset-0 w-full h-full object-cover',
            annotatedFrameUrl && isRunning ? 'opacity-0' : 'opacity-100'
          )}
          muted
          playsInline
          autoPlay
        />

        {/* Annotated frame from backend (face mesh overlay) */}
        {annotatedFrameUrl && isRunning && (
          <img
            src={annotatedFrameUrl}
            alt="Annotated face analysis"
            className="absolute inset-0 w-full h-full object-cover"
            style={{ imageRendering: 'pixelated' }}
          />
        )}

        {/* Idle state */}
        {!isRunning && !cameraError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-dark-900/80 backdrop-blur-sm">
            <div className="w-16 h-16 rounded-full border-2 border-brand-500/30 flex items-center justify-center mb-4 animate-pulse-slow">
              <Camera size={28} className="text-brand-500/60" />
            </div>
            <p className="text-dark-300 text-sm font-mono">Camera inactive</p>
          </div>
        )}

        {/* Error state */}
        {cameraError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-dark-900/90 p-4">
            <CameraOff size={32} className="text-risk-critical mb-3" />
            <p className="text-risk-critical text-sm font-mono text-center">{cameraError}</p>
          </div>
        )}

        {/* Face not detected overlay */}
        {isRunning && latestResult && !latestResult.face_detected && (
          <div className="absolute bottom-3 left-3 right-3">
            <div className="bg-yellow-500/20 border border-yellow-500/40 rounded-lg px-3 py-2 flex items-center gap-2">
              <AlertTriangle size={14} className="text-yellow-400 flex-shrink-0" />
              <span className="text-yellow-300 text-xs font-mono">No face detected — please center your face</span>
            </div>
          </div>
        )}

        {/* Alert banners */}
        {activeAlerts.length > 0 && (
          <div className="absolute top-3 left-3 right-3 space-y-1">
            {activeAlerts.slice(0, 2).map((alert, i) => (
              <div key={i} className="bg-risk-critical/20 border border-risk-critical/40 rounded-lg px-3 py-2 animate-slide-in-right">
                <span className="text-risk-critical text-xs font-mono">{alert}</span>
              </div>
            ))}
          </div>
        )}

        {/* Corner scan effect */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-brand-500/60 rounded-tl-lg" />
          <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-brand-500/60 rounded-tr-lg" />
          <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-brand-500/60 rounded-bl-lg" />
          <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-brand-500/60 rounded-br-lg" />
        </div>

        {/* Landmark count badge */}
        {isRunning && latestResult?.face_detected && (
          <div className="absolute bottom-3 right-3">
            <span className="text-xs font-mono bg-dark-800/80 text-brand-500 border border-brand-500/30 px-2 py-0.5 rounded">
              {latestResult.landmark_count} landmarks
            </span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="mt-3 flex gap-2">
        {!isRunning ? (
          <button
            id="btn-start-analysis"
            onClick={start}
            className="flex-1 bg-brand-500 hover:bg-brand-400 text-dark-950 font-bold py-2.5 px-4 rounded-lg
                       transition-all duration-200 flex items-center justify-center gap-2 text-sm
                       shadow-[0_0_20px_rgba(0,255,65,0.3)] hover:shadow-[0_0_30px_rgba(0,255,65,0.5)]"
          >
            <Camera size={16} />
            Start Analysis
          </button>
        ) : (
          <button
            id="btn-stop-analysis"
            onClick={stop}
            className="flex-1 bg-risk-critical/20 hover:bg-risk-critical/30 border border-risk-critical/40 
                       text-risk-critical font-semibold py-2.5 px-4 rounded-lg transition-all duration-200
                       flex items-center justify-center gap-2 text-sm"
          >
            <CameraOff size={16} />
            Stop Analysis
          </button>
        )}
      </div>
    </div>
  );
};
