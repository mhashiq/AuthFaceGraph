/**
 * AuthBrain AI Face Analysis Engine
 * TypeScript Type Definitions
 *
 * Matches backend Pydantic schemas exactly.
 * All WebSocket and REST response shapes are defined here.
 */

// ══════════════════════════════════════════════════════════════════════════════
// Enums
// ══════════════════════════════════════════════════════════════════════════════

export type AttentionState = 'focused' | 'distracted' | 'drowsy' | 'alert' | 'unknown';
export type GazeDirection  = 'center' | 'left' | 'right' | 'up' | 'down' | 'closed';
export type SessionStatus  = 'active' | 'paused' | 'completed' | 'error';
export type UserRole       = 'employee' | 'manager' | 'researcher' | 'administrator';
export type RiskLevel      = 'low' | 'medium' | 'high' | 'critical';

// ══════════════════════════════════════════════════════════════════════════════
// Landmark & Geometry
// ══════════════════════════════════════════════════════════════════════════════

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface FaceBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// ══════════════════════════════════════════════════════════════════════════════
// Analysis Results
// ══════════════════════════════════════════════════════════════════════════════

export interface HeadPoseResult {
  pitch: number;
  yaw: number;
  roll: number;
  is_facing_forward: boolean;
}

export interface EyeResult {
  ear: number;
  is_open: boolean;
  gaze_x: number;
  gaze_y: number;
}

export interface EyeAnalysisResult {
  left: EyeResult;
  right: EyeResult;
  average_ear: number;
  blink_detected: boolean;
  blink_count: number;
  eye_closure_duration_ms: number;
  gaze_direction: GazeDirection;
  blinks_per_minute: number;
}

export interface MouthAnalysisResult {
  mar: number;
  is_open: boolean;
  yawn_detected: boolean;
  yawn_confidence: number;
  smile_intensity: number;
  mouth_openness_percent: number;
}

export interface BehaviorResult {
  head_movement_velocity: number;
  facial_movement_score: number;
  landmark_stability: number;
  facial_symmetry: number;
  attention_state: AttentionState;
}

export interface QualityResult {
  overall_score: number;
  sharpness: number;
  illumination: number;
  face_size_ratio: number;
  landmark_confidence: number;
}

// ══════════════════════════════════════════════════════════════════════════════
// XAI & Expert System
// ══════════════════════════════════════════════════════════════════════════════

export interface FeatureAttribution {
  feature_name: string;
  contribution: number;
  landmark_indices: number[];
  value: number;
  description: string;
}

export interface ExplanationResult {
  metric_name: string;
  final_value: number;
  confidence: number;
  attributions: FeatureAttribution[];
  processing_time_ms: number;
  landmark_quality: number;
  explanation_text: string;
}

export interface ExpertSystemResult {
  attention_state: AttentionState;
  fatigue_score: number;
  focus_score: number;
  alerts: string[];
  explanations: ExplanationResult[];
  overall_confidence: number;
}

// ══════════════════════════════════════════════════════════════════════════════
// Main Per-Frame Result (WebSocket payload)
// ══════════════════════════════════════════════════════════════════════════════

export interface FaceAnalysisResult {
  frame_id: string;
  session_id: string;
  timestamp: string;
  inference_time_ms: number;
  face_detected: boolean;
  face_count: number;
  active_face_index: number;
  bounding_box: FaceBoundingBox | null;
  landmark_count: number;
  head_pose: HeadPoseResult | null;
  eyes: EyeAnalysisResult | null;
  mouth: MouthAnalysisResult | null;
  behavior: BehaviorResult | null;
  quality: QualityResult | null;
  expert_system: ExpertSystemResult | null;
  deep_learning: DLAnalysisResult | null;
  fps: number;
  frame_width: number;
  frame_height: number;
  model_confidence: number;
}

// ══════════════════════════════════════════════════════════════════════════════
// Deep Learning Interfaces
// ══════════════════════════════════════════════════════════════════════════════

export type EmotionLabel = 'neutral' | 'happy' | 'sad' | 'surprise' | 'fear' | 'disgust' | 'anger' | 'contempt';

export interface DLEmotionPrediction {
  emotion: EmotionLabel;
  confidence: number;
  probabilities: Record<string, number>;
  model_id: string;
  latency_ms: number;
}

export interface ActionUnit {
  au_id: string;
  name: string;
  present: boolean;
  intensity: number;
}

export interface GNNPrediction {
  emotion: EmotionLabel;
  confidence: number;
  probabilities: Record<string, number>;
  model_id: string;
  node_importance: number[];
  edge_attention: number[];
}

export interface EnsembleResult {
  final_emotion: EmotionLabel;
  confidence: number;
  probabilities: Record<string, number>;
  model_predictions: DLEmotionPrediction[];
  disagreement_score: number;
  uncertainty: number;
}

export interface DLAnalysisResult {
  dl_enabled: boolean;
  dl_inference_time_ms: number;
  emotion_ensemble: EnsembleResult | null;
  gnn_prediction: GNNPrediction | null;
  action_units: ActionUnit[];
  top_important_landmarks: number[];
  models_used: string[];
  xai_explanations: ExplanationResult[];
  landmarks: Landmark[];
}

// ══════════════════════════════════════════════════════════════════════════════
// WebSocket Messages
// ══════════════════════════════════════════════════════════════════════════════

export type WSMessageType = 'analysis_result' | 'error' | 'status' | 'config' | 'ping';

export interface WSMessage<T = unknown> {
  type: WSMessageType;
  payload: T;
  timestamp: string;
}

export interface WSStatusPayload {
  status: string;
  session_id: string;
  user_id: string;
  fps_target: number;
}

export interface WSErrorPayload {
  code: string;
  message: string;
  detail?: string;
}

// ══════════════════════════════════════════════════════════════════════════════
// Auth & User
// ══════════════════════════════════════════════════════════════════════════════

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface LoginResponse {
  tokens: TokenPair;
  user_id: string;
  email: string;
  role: UserRole;
  org_id: string;
  full_name: string;
}

export interface ConsentResponse {
  session_id: string;
  consent_granted: boolean;
  analysis_token: string;
  expires_at: string;
  message: string;
}

// ══════════════════════════════════════════════════════════════════════════════
// Health
// ══════════════════════════════════════════════════════════════════════════════

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  database_connected: boolean;
  model_loaded: boolean;
  gpu_available: boolean;
  uptime_seconds: number;
  active_sessions: number;
}

// ══════════════════════════════════════════════════════════════════════════════
// Chart data helpers
// ══════════════════════════════════════════════════════════════════════════════

export interface EARDataPoint {
  time: number;
  left_ear: number;
  right_ear: number;
  avg_ear: number;
  threshold: number;
}

export interface HeadPoseDataPoint {
  time: number;
  pitch: number;
  yaw: number;
  roll: number;
}

export interface FatigueDataPoint {
  time: number;
  fatigue: number;
  focus: number;
}

export interface SystemLogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error';
  message: string;
  source: string;
}
