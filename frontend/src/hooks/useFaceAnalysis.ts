/**
 * AuthBrain AI Face Analysis Engine
 * useFaceAnalysis Hook
 *
 * Orchestrates the full analysis pipeline on the client side:
 * 1. Opens webcam via getUserMedia
 * 2. Captures frames from video element at TARGET_FPS
 * 3. Sends JPEG frames to backend via WebSocket
 * 4. Receives and stores analysis results and annotated frames
 *
 * Usage:
 *   const { videoRef, isRunning, start, stop } = useFaceAnalysis();
 */

import { useRef, useCallback, useEffect, useState } from 'react';
import { FaceAnalysisWSService } from '../services/websocket';
import { useAnalysisStore, useAuthStore } from '../store';
import type { FaceAnalysisResult } from '../types/analysis';

const TARGET_FPS  = 30;
const FRAME_DELAY = 1000 / TARGET_FPS;
const JPEG_QUALITY = 0.8;

export function useFaceAnalysis() {
  const videoRef     = useRef<HTMLVideoElement>(null);
  const streamRef    = useRef<MediaStream | null>(null);
  const wsService    = useRef<FaceAnalysisWSService | null>(null);
  const frameLoopRef = useRef<number | null>(null);
  const lastFrameTime   = useRef<number>(0);
  const isProcessingRef = useRef<boolean>(false);

  // Off-screen canvas kept in a ref so it is never recreated on re-renders
  // and avoids module-level side effects (which break HMR).
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureCtxRef    = useRef<CanvasRenderingContext2D | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const { accessToken, userId } = useAuthStore();

  // ── Pull store actions via getState so we get stable references ────────────
  // Calling useAnalysisStore() would return new function references on every
  // render triggered by a log push, causing the cleanup useEffect to re-run
  // and call stop() immediately — the root cause of the camera crash.
  const storeRef = useRef(useAnalysisStore.getState());
  useEffect(() => {
    // Keep storeRef in sync if store is swapped (rare but possible in tests)
    storeRef.current = useAnalysisStore.getState();
  });

  // ── Frame capture loop ────────────────────────────────────────────────────
  const captureAndSend = useCallback((timestamp: number) => {
    const video   = videoRef.current;
    const service = wsService.current;

    if (!video || !service || !service.isConnected) {
      frameLoopRef.current = requestAnimationFrame(captureAndSend);
      return;
    }

    // Skip if previous frame is still in-flight to prevent bufferbloat
    if (isProcessingRef.current) {
      frameLoopRef.current = requestAnimationFrame(captureAndSend);
      return;
    }

    // Throttle to TARGET_FPS
    if (timestamp - lastFrameTime.current < FRAME_DELAY) {
      frameLoopRef.current = requestAnimationFrame(captureAndSend);
      return;
    }
    lastFrameTime.current = timestamp;

    // Draw video frame to off-screen canvas
    const { videoWidth, videoHeight } = video;
    if (videoWidth === 0 || videoHeight === 0) {
      frameLoopRef.current = requestAnimationFrame(captureAndSend);
      return;
    }

    // Lazily initialise the capture canvas inside the hook (avoids module-level
    // side effects and is safe with HMR / SSR).
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas');
      captureCtxRef.current    = captureCanvasRef.current.getContext('2d');
    }
    const canvas = captureCanvasRef.current;
    const ctx    = captureCtxRef.current;
    if (!canvas || !ctx) {
      frameLoopRef.current = requestAnimationFrame(captureAndSend);
      return;
    }

    canvas.width  = videoWidth;
    canvas.height = videoHeight;
    ctx.drawImage(video, 0, 0);

    isProcessingRef.current = true;

    // Convert to JPEG blob and send
    canvas.toBlob(
      (blob) => {
        if (blob && service.isConnected) {
          service.sendFrame(blob);
        } else {
          isProcessingRef.current = false;
        }
      },
      'image/jpeg',
      JPEG_QUALITY,
    );

    frameLoopRef.current = requestAnimationFrame(captureAndSend);
  }, []); // stable — relies only on refs

  // ── Internal stop (does NOT depend on addLog so it stays stable) ──────────
  const stopInternal = useCallback(() => {
    // Cancel frame loop
    if (frameLoopRef.current !== null) {
      cancelAnimationFrame(frameLoopRef.current);
      frameLoopRef.current = null;
    }

    // Disconnect WebSocket
    wsService.current?.disconnect();
    wsService.current = null;

    // Stop camera stream
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    isProcessingRef.current = false;
  }, []); // ← no dependencies → never recreated → cleanup useEffect won't loop

  // ── Start analysis ────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    const { accessToken: token, sessionId } = {
      accessToken,
      sessionId: storeRef.current.sessionId,
    };

    if (!token || !sessionId) {
      setCameraError('Authentication required. Please log in and grant consent.');
      return;
    }

    // Reset session store values before starting to clear stale landmarks/metrics
    storeRef.current.resetSession();

    // Request camera access
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:     { ideal: 1280 },
          height:    { ideal: 720  },
          frameRate: { ideal: TARGET_FPS },
          facingMode: 'user',
        },
        audio: false,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Camera access denied';
      setCameraError(msg);
      storeRef.current.addLog('error', `Camera error: ${msg}`, 'camera');
      return;
    }

    // Initialize WebSocket service
    wsService.current = new FaceAnalysisWSService({
      onResult: (result: FaceAnalysisResult) => {
        isProcessingRef.current = false;
        storeRef.current.pushResult(result);
      },
      onAnnotatedFrame: (blob: Blob) => {
        const url = URL.createObjectURL(blob);
        storeRef.current.setAnnotatedFrame(url);
      },
      onStatusChange: (state) => {
        storeRef.current.setWsState(state);
        storeRef.current.addLog('info', `WebSocket: ${state}`, 'websocket');
      },
      onAlert: (alerts) => {
        storeRef.current.setAlerts(alerts);
        alerts.forEach(a => storeRef.current.addLog('warning', a, 'expert-system'));
      },
      onError: (message) => {
        isProcessingRef.current = false;
        storeRef.current.addLog('error', message, 'websocket');
      },
    });

    wsService.current.connect(token, sessionId);

    // Start frame capture loop
    frameLoopRef.current = requestAnimationFrame(captureAndSend);
    setIsRunning(true);
    storeRef.current.addLog('info', 'Analysis session started', 'pipeline');
  }, [accessToken, captureAndSend]);

  // ── Public stop (adds log, then delegates to stopInternal) ───────────────
  const stop = useCallback(() => {
    stopInternal();
    setIsRunning(false);
    // Read addLog from store directly to avoid capturing it in deps
    useAnalysisStore.getState().addLog('info', 'Analysis session stopped', 'pipeline');
  }, [stopInternal]);

  // ── Cleanup on unmount ────────────────────────────────────────────────────
  // IMPORTANT: Only stopInternal (which is stable / never recreated) is used
  // here. Using the public `stop` or any function that depends on `addLog`
  // would cause the cleanup to re-run on every log push, instantly killing
  // the camera stream each time a result arrives.
  useEffect(() => {
    return () => {
      stopInternal();
      setIsRunning(false);
    };
  }, [stopInternal]); // stopInternal is stable (empty deps), so this runs exactly once

  return {
    videoRef,
    isRunning,
    cameraError,
    start,
    stop,
  };
}
