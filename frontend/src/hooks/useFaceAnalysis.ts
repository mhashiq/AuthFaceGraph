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

// Canvas used for frame capture (off-screen)
const _captureCanvas = document.createElement('canvas');
const _captureCtx    = _captureCanvas.getContext('2d')!;


export function useFaceAnalysis() {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsService = useRef<FaceAnalysisWSService | null>(null);
  const frameLoopRef = useRef<number | null>(null);
  const lastFrameTime = useRef<number>(0);

  const [isRunning, setIsRunning] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const isProcessingRef = useRef<boolean>(false);

  const { accessToken, userId } = useAuthStore();
  const {
    sessionId,
    pushResult,
    setAnnotatedFrame,
    setWsState,
    setAlerts,
    addLog,
  } = useAnalysisStore();

  // ── Frame capture loop ────────────────────────────────────────────────────
  const captureAndSend = useCallback((timestamp: number) => {
    const video = videoRef.current;
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

    _captureCanvas.width  = videoWidth;
    _captureCanvas.height = videoHeight;
    _captureCtx.drawImage(video, 0, 0);

    isProcessingRef.current = true;

    // Convert to JPEG blob and send
    _captureCanvas.toBlob(
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
  }, []);

  // ── Start analysis ────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    if (!accessToken || !sessionId) {
      setCameraError('Authentication required. Please log in and grant consent.');
      return;
    }

    // Request camera access
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:  { ideal: 1280 },
          height: { ideal: 720 },
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
      addLog('error', `Camera error: ${msg}`, 'camera');
      return;
    }

    // Initialize WebSocket service
    wsService.current = new FaceAnalysisWSService({
      onResult: (result: FaceAnalysisResult) => {
        isProcessingRef.current = false;
        pushResult(result);
      },
      onAnnotatedFrame: (blob: Blob) => {
        const url = URL.createObjectURL(blob);
        setAnnotatedFrame(url);
      },
      onStatusChange: (state) => {
        setWsState(state);
        addLog('info', `WebSocket: ${state}`, 'websocket');
      },
      onAlert: (alerts) => {
        setAlerts(alerts);
        alerts.forEach(a => addLog('warning', a, 'expert-system'));
      },
      onError: (message) => {
        isProcessingRef.current = false;
        addLog('error', message, 'websocket');
      },
    });

    wsService.current.connect(accessToken, sessionId);

    // Start frame capture loop
    frameLoopRef.current = requestAnimationFrame(captureAndSend);
    setIsRunning(true);
    addLog('info', 'Analysis session started', 'pipeline');
  }, [accessToken, sessionId, captureAndSend, pushResult, setAnnotatedFrame, setWsState, setAlerts, addLog]);

  // ── Stop analysis ─────────────────────────────────────────────────────────
  const stop = useCallback(() => {
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

    setIsRunning(false);
    addLog('info', 'Analysis session stopped', 'pipeline');
  }, [addLog]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    videoRef,
    isRunning,
    cameraError,
    start,
    stop,
  };
}
