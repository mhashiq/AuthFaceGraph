/**
 * AuthBrain AI Face Analysis Engine
 * Zustand Global State Store
 *
 * Centralized state for:
 * - Authentication (JWT tokens, user info)
 * - Consent state
 * - Analysis results (latest + rolling history)
 * - WebSocket connection state
 * - System logs
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type {
  FaceAnalysisResult,
  LoginResponse,
  EARDataPoint,
  HeadPoseDataPoint,
  FatigueDataPoint,
  SystemLogEntry,
} from '../types/analysis';
import type { WSConnectionState } from '../services/websocket';

const CHART_HISTORY_POINTS = 120;  // 4 seconds at 30fps

// ══════════════════════════════════════════════════════════════════════════════
// Auth Store
// ══════════════════════════════════════════════════════════════════════════════

interface AuthState {
  accessToken:  string | null;
  refreshToken: string | null;
  userId:       string | null;
  email:        string | null;
  role:         string | null;
  orgId:        string | null;
  fullName:     string | null;
  isAuthenticated: boolean;

  setAuth:    (response: LoginResponse) => void;
  clearAuth:  () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      accessToken:     null,
      refreshToken:    null,
      userId:          null,
      email:           null,
      role:            null,
      orgId:           null,
      fullName:        null,
      isAuthenticated: false,

      setAuth: (response: LoginResponse) => set({
        accessToken:     response.tokens.access_token,
        refreshToken:    response.tokens.refresh_token,
        userId:          response.user_id,
        email:           response.email,
        role:            response.role,
        orgId:           response.org_id,
        fullName:        response.full_name,
        isAuthenticated: true,
      }),

      clearAuth: () => set({
        accessToken: null, refreshToken: null, userId: null,
        email: null, role: null, orgId: null, fullName: null,
        isAuthenticated: false,
      }),
    }),
    {
      name:    'authbrain-auth',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        accessToken:     state.accessToken,
        refreshToken:    state.refreshToken,
        userId:          state.userId,
        email:           state.email,
        role:            state.role,
        orgId:           state.orgId,
        fullName:        state.fullName,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

// ══════════════════════════════════════════════════════════════════════════════
// Analysis State Store
// ══════════════════════════════════════════════════════════════════════════════

interface AnalysisState {
  // Connection
  wsState:   WSConnectionState;
  sessionId: string | null;
  consentGranted: boolean;

  // Latest frame result
  latestResult: FaceAnalysisResult | null;
  annotatedFrameUrl: string | null;  // Object URL of latest annotated JPEG

  // Rolling chart data
  earHistory:     EARDataPoint[];
  headPoseHistory: HeadPoseDataPoint[];
  fatigueHistory: FatigueDataPoint[];
  emotionHistory: {
    time: number;
    happy: number;
    sad: number;
    neutral: number;
    surprise: number;
    anger: number;
  }[];

  // Alerts
  activeAlerts: string[];
  alertExpirations: Record<string, number>;
  undetectedCount: number;

  // System logs
  logs: SystemLogEntry[];

  // Computed
  sessionBlinkCount:  number;
  sessionStartTime:   number | null;

  // Actions
  setWsState:        (state: WSConnectionState) => void;
  setSessionId:      (id: string) => void;
  setConsentGranted: (granted: boolean) => void;
  pushResult:        (result: FaceAnalysisResult) => void;
  setAnnotatedFrame: (url: string) => void;
  setAlerts:         (alerts: string[]) => void;
  addLog:            (level: 'info' | 'warning' | 'error', message: string, source?: string) => void;
  resetSession:      () => void;
}

let logIdCounter = 0;

export const useAnalysisStore = create<AnalysisState>()((set, get) => ({
  wsState:          'disconnected',
  sessionId:        null,
  consentGranted:   false,
  latestResult:     null,
  annotatedFrameUrl: null,
  earHistory:       [],
  headPoseHistory:  [],
  fatigueHistory:   [],
  emotionHistory:   [],
  activeAlerts:     [],
  alertExpirations: {},
  undetectedCount:  0,
  logs:             [],
  sessionBlinkCount: 0,
  sessionStartTime:  null,

  setWsState: (state) => set({ wsState: state }),
  setSessionId: (id) => set({ sessionId: id }),
  setConsentGranted: (granted) => set({ consentGranted: granted }),

  pushResult: (result: FaceAnalysisResult) => {
    const now = Date.now();
    const state = get();

    // 1. Smooth face detection drops using a 15-frame buffer (approx 500ms)
    let undetectedCount = state.undetectedCount;
    let faceDetected = result.face_detected;
    if (!faceDetected) {
      undetectedCount += 1;
      if (undetectedCount < 15) {
        faceDetected = true;
      }
    } else {
      undetectedCount = 0;
    }

    const smoothedResult = {
      ...result,
      face_detected: faceDetected,
    };

    // 2. Clean up expired alerts
    const updatedExpirations = { ...state.alertExpirations };
    let expirationsChanged = false;
    Object.keys(updatedExpirations).forEach((alert) => {
      if (updatedExpirations[alert] < now) {
        delete updatedExpirations[alert];
        expirationsChanged = true;
      }
    });

    const activeAlerts = Object.keys(updatedExpirations);

    // Update chart histories
    const time = now;

    const newEarPoint: EARDataPoint | null = result.eyes ? {
      time,
      left_ear:  result.eyes.left.ear,
      right_ear: result.eyes.right.ear,
      avg_ear:   result.eyes.average_ear,
      threshold: 0.25,
    } : null;

    const newPosePoint: HeadPoseDataPoint | null = result.head_pose ? {
      time,
      pitch: result.head_pose.pitch,
      yaw:   result.head_pose.yaw,
      roll:  result.head_pose.roll,
    } : null;

    const newFatiguePoint: FatigueDataPoint | null = result.expert_system ? {
      time,
      fatigue: result.expert_system.fatigue_score,
      focus:   result.expert_system.focus_score,
    } : null;

    const dl = result.deep_learning;
    const newEmotionPoint = dl && dl.emotion_ensemble ? {
      time,
      happy: dl.emotion_ensemble.probabilities['happy'] ?? 0,
      sad: dl.emotion_ensemble.probabilities['sad'] ?? 0,
      neutral: dl.emotion_ensemble.probabilities['neutral'] ?? 0,
      surprise: dl.emotion_ensemble.probabilities['surprise'] ?? 0,
      anger: dl.emotion_ensemble.probabilities['anger'] ?? 0,
    } : null;

    set({
      latestResult: smoothedResult,
      undetectedCount,
      alertExpirations: updatedExpirations,
      activeAlerts: activeAlerts,
      sessionBlinkCount: result.eyes?.blink_count ?? state.sessionBlinkCount,
      sessionStartTime:  state.sessionStartTime ?? now,

      earHistory: newEarPoint
        ? [...state.earHistory.slice(-CHART_HISTORY_POINTS + 1), newEarPoint]
        : state.earHistory,

      headPoseHistory: newPosePoint
        ? [...state.headPoseHistory.slice(-CHART_HISTORY_POINTS + 1), newPosePoint]
        : state.headPoseHistory,

      fatigueHistory: newFatiguePoint
        ? [...state.fatigueHistory.slice(-CHART_HISTORY_POINTS + 1), newFatiguePoint]
        : state.fatigueHistory,

      emotionHistory: newEmotionPoint
        ? [...state.emotionHistory.slice(-CHART_HISTORY_POINTS + 1), newEmotionPoint]
        : state.emotionHistory,
    });
  },

  setAnnotatedFrame: (url: string) => {
    // Revoke previous Object URL to prevent memory leak
    const prev = get().annotatedFrameUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({ annotatedFrameUrl: url });
  },

  setAlerts: (newAlerts: string[]) => {
    const now = Date.now();
    const state = get();
    const updatedExpirations = { ...state.alertExpirations };
    let changed = false;

    newAlerts.forEach((alert) => {
      // Keep each alert visible for at least 3 seconds from its last trigger
      updatedExpirations[alert] = now + 3000;
      changed = true;
    });

    if (changed || newAlerts.length === 0) {
      set({
        alertExpirations: updatedExpirations,
        activeAlerts: Object.keys(updatedExpirations),
      });
    }
  },

  addLog: (level, message, source = 'system') => {
    const entry: SystemLogEntry = {
      id:        String(++logIdCounter),
      timestamp: new Date().toISOString(),
      level,
      message,
      source,
    };
    set((s) => ({ logs: [entry, ...s.logs].slice(0, 200) }));
  },

  resetSession: () => set({
    latestResult:      null,
    annotatedFrameUrl: null,
    earHistory:        [],
    headPoseHistory:   [],
    fatigueHistory:    [],
    emotionHistory:    [],
    activeAlerts:      [],
    sessionBlinkCount: 0,
    sessionStartTime:  null,
  }),
}));
