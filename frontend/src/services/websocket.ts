/**
 * AuthBrain AI Face Analysis Engine
 * WebSocket Service
 *
 * Manages WebSocket connection lifecycle:
 * - Connects with JWT token authentication
 * - Sends JPEG frames as binary ArrayBuffer
 * - Receives JSON analysis results and binary annotated frames
 * - Handles reconnection with exponential backoff
 */

import type { FaceAnalysisResult, WSMessage, WSStatusPayload, WSErrorPayload } from '../types/analysis';

export type WSConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WSServiceCallbacks {
  onResult:      (result: FaceAnalysisResult) => void;
  onAnnotatedFrame: (imageBlob: Blob) => void;
  onStatusChange: (state: WSConnectionState) => void;
  onAlert:        (alerts: string[]) => void;
  onError:        (message: string) => void;
}

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY_MS = 1000;


export class FaceAnalysisWSService {
  private ws: WebSocket | null = null;
  private token: string = '';
  private sessionId: string = '';
  private callbacks: WSServiceCallbacks;
  private reconnectAttempts: number = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private isIntentionalClose: boolean = false;

  // Binary message routing: alternating JSON result → binary frame
  private expectingBinary: boolean = false;

  constructor(callbacks: WSServiceCallbacks) {
    this.callbacks = callbacks;
  }

  connect(token: string, sessionId: string): void {
    this.token = token;
    this.sessionId = sessionId;
    this.isIntentionalClose = false;
    this._connect();
  }

  private _connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    const url = `${WS_BASE_URL}/ws/analyze?token=${encodeURIComponent(this.token)}&session_id=${this.sessionId}`;

    this.callbacks.onStatusChange('connecting');
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';  // Efficient binary frame reception

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.callbacks.onStatusChange('connected');
    };

    this.ws.onclose = (event) => {
      if (!this.isIntentionalClose) {
        this.callbacks.onStatusChange('disconnected');
        this._scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      this.callbacks.onStatusChange('error');
      this.callbacks.onError('WebSocket connection error');
    };

    this.ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        // JSON message (analysis result or status)
        this._handleJsonMessage(event.data);
      } else if (event.data instanceof ArrayBuffer) {
        // Binary message (annotated JPEG frame)
        this._handleBinaryMessage(event.data);
      }
    };
  }

  private _handleJsonMessage(raw: string): void {
    try {
      const msg = JSON.parse(raw) as WSMessage;

      switch (msg.type) {
        case 'analysis_result': {
          const result = msg.payload as FaceAnalysisResult;
          this.callbacks.onResult(result);
          if (result.expert_system?.alerts?.length) {
            this.callbacks.onAlert(result.expert_system.alerts);
          }
          break;
        }
        case 'status': {
          const status = msg.payload as WSStatusPayload;
          console.info('[AuthBrain WS] Status:', status);
          break;
        }
        case 'error': {
          const errPayload = msg.payload as WSErrorPayload;
          this.callbacks.onError(errPayload.message || 'Unknown error');
          break;
        }
        case 'ping':
          // Pong response not needed for browser WebSocket
          break;
      }
    } catch (err) {
      console.error('[AuthBrain WS] JSON parse error:', err);
    }
  }

  private _handleBinaryMessage(buffer: ArrayBuffer): void {
    const blob = new Blob([buffer], { type: 'image/jpeg' });
    this.callbacks.onAnnotatedFrame(blob);
  }

  sendFrame(jpegBlob: Blob): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    // Send as ArrayBuffer for best performance
    jpegBlob.arrayBuffer().then((buffer) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(buffer);
      }
    });
  }

  private _scheduleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      this.callbacks.onError('Max reconnection attempts reached');
      return;
    }

    const delay = BASE_RECONNECT_DELAY_MS * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    this.reconnectTimer = setTimeout(() => {
      this._connect();
    }, delay);
  }

  disconnect(): void {
    this.isIntentionalClose = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    if (this.ws) {
      this.ws.close(1000, 'Client initiated disconnect');
      this.ws = null;
    }
    this.callbacks.onStatusChange('disconnected');
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
