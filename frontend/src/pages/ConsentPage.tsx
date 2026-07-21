/**
 * AuthFaceGraph — Premium Consent, Login & Registration Page
 * Cinematic dark glassmorphic authentication & biometric consent workspace.
 */

import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff, Shield, CheckCircle, ChevronRight, Zap, Lock, UserPlus, UserCheck } from 'lucide-react';
import axios from 'axios';
import { useAuthStore, useAnalysisStore } from '../store';
import type { LoginResponse, ConsentResponse } from '../types/analysis';
import { NeonButton } from '../components/ui';
import { FaceEnrollmentWizard } from '../components/auth/FaceEnrollmentWizard';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const ConsentPage: React.FC = () => {
  const navigate    = useNavigate();
  const { setAuth } = useAuthStore();
  const { setConsentGranted, setSessionId, addLog } = useAnalysisStore();

  const [step, setStep]         = useState<'login' | 'enrollment' | 'consent'>('login');
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  
  const [fullName, setFullName] = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [orgId, setOrgId]       = useState('');
  
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [consentChecked, setConsentChecked] = useState(false);
  const [accessToken, setAccessToken]       = useState('');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animated background canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const particles: Array<{x:number;y:number;vx:number;vy:number;size:number;opacity:number;color:string}> = [];
    const COLORS = ['#00d4ff', '#7c3aed', '#4f46e5', '#2563eb'];
    for (let i = 0; i < 80; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: Math.random() * 2 + 0.5,
        opacity: Math.random() * 0.5 + 0.1,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
      });
    }

    let raf = 0;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Background
      const grad = ctx.createRadialGradient(canvas.width/2, canvas.height/2, 0, canvas.width/2, canvas.height/2, canvas.width * 0.8);
      grad.addColorStop(0, '#0a0f1e');
      grad.addColorStop(1, '#010409');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Grid
      ctx.strokeStyle = 'rgba(0,212,255,0.04)';
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      // Particles + connections
      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color + Math.floor(p.opacity * 255).toString(16).padStart(2,'0');
        ctx.shadowBlur = 8;
        ctx.shadowColor = p.color;
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx*dx + dy*dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(79,70,229,${(1 - dist/120) * 0.12})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
  }, []);

  const fillDemoAdmin = () => {
    setEmail('admin@authbrain.com');
    setPassword('password123');
    setAuthMode('signin');
    setError(null);
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post<LoginResponse>(`${API_BASE}/api/auth/login`, {
        email: email.trim(),
        password,
        org_id: orgId.trim() || undefined,
      });
      setAuth(res.data);
      setAccessToken(res.data.tokens.access_token);
      setStep('enrollment');
      addLog('info', `Logged in as ${res.data.email}`, 'auth');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Authentication failed. Check credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post<LoginResponse>(`${API_BASE}/api/auth/register`, {
        full_name: fullName.trim() || 'AuthBrain User',
        email: email.trim(),
        password,
        organization_name: 'AuthBrain Enterprise',
      });
      setAuth(res.data);
      setAccessToken(res.data.tokens.access_token);
      setStep('enrollment');
      addLog('info', `Registered new account: ${res.data.email}`, 'auth');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Registration failed. Try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleConsent = async () => {
    if (!consentChecked) return;
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post<ConsentResponse>(
        `${API_BASE}/api/consent/`,
        { consent_granted: true, purpose: 'biometric_analysis' },
        { headers: { Authorization: `Bearer ${accessToken}` } },
      );
      setConsentGranted(true);
      setSessionId(res.data.session_id);
      addLog('info', 'Consent granted — analysis session started', 'consent');
      navigate('/dashboard', { replace: true });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Consent registration failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden gpu-accelerated py-10">
      {/* Animated BG canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 z-0" />

      {/* Ambient orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full opacity-10 animate-glow-pulse"
        style={{ background: 'radial-gradient(circle, #7c3aed, transparent 70%)', filter: 'blur(60px)' }} />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full opacity-10 animate-glow-pulse"
        style={{ background: 'radial-gradient(circle, #00d4ff, transparent 70%)', filter: 'blur(60px)', animationDelay: '1s' }} />

      {/* Brand top-left */}
      <div className="absolute top-6 left-8 z-10 flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: 'linear-gradient(135deg,#4f46e5,#7c3aed)', boxShadow: '0 0 16px rgba(139,92,246,0.4)' }}>
          <Shield size={16} className="text-white" />
        </div>
        <span className="font-display font-bold text-sm tracking-widest text-gradient-brand uppercase">
          AuthFaceGraph
        </span>
      </div>

      {/* System status top-right */}
      <div className="absolute top-6 right-8 z-10 flex items-center gap-1.5">
        <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" style={{ boxShadow: '0 0 6px #10b981' }} />
        <span className="font-mono text-[10px] text-emerald-400/70 uppercase tracking-widest">Systems Online</span>
      </div>

      {/* ── MAIN CARD ──────────────────────────────────────────────── */}
      <div className="relative z-10 w-full max-w-md mx-4 animate-card-enter">
        <div className="gradient-border p-[1px] rounded-2xl"
          style={{ background: 'linear-gradient(135deg, rgba(0,212,255,0.4), rgba(79,70,229,0.4), rgba(139,92,246,0.4))' }}>
          <div className="rounded-2xl p-8"
            style={{ background: 'rgba(7,13,26,0.94)', backdropFilter: 'blur(24px)' }}>

            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
              <div className="w-11 h-11 rounded-xl flex items-center justify-center"
                style={{ background: 'linear-gradient(135deg,#4f46e5,#7c3aed)', boxShadow: '0 0 20px rgba(139,92,246,0.5)' }}>
                {step === 'login' ? (authMode === 'signin' ? <Lock size={20} className="text-white" /> : <UserPlus size={20} className="text-white" />) : <Shield size={20} className="text-white" />}
              </div>
              <div>
                <h1 className="text-xl font-bold text-white font-display tracking-wide">
                  {step === 'login' ? (authMode === 'signin' ? 'Secure Sign In' : 'Create Account') : 'Biometric Consent'}
                </h1>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  {step === 'login' ? 'AuthFaceGraph AI Platform' : 'Review & accept data processing terms'}
                </p>
              </div>
            </div>

            {/* Step indicator */}
            <div className="flex items-center gap-1.5 mb-6 overflow-x-auto">
              {['login', 'enrollment', 'consent'].map((s, i) => (
                <React.Fragment key={s}>
                  <div className="flex items-center gap-1">
                    <div className="w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold font-mono"
                      style={{
                        background: step === s
                          ? 'linear-gradient(135deg,#4f46e5,#7c3aed)'
                          : 'rgba(79,70,229,0.1)',
                        color: step === s ? 'white' : '#5d7399',
                        border: `1px solid ${step === s ? 'rgba(139,92,246,0.6)' : 'rgba(79,70,229,0.2)'}`,
                      }}>
                      {i + 1}
                    </div>
                    <span className="font-mono text-[9px] uppercase tracking-wider"
                      style={{ color: step === s ? '#c9d4f0' : '#3a4f70' }}>
                      {s === 'login' ? 'Auth' : s === 'enrollment' ? 'Enroll' : 'Consent'}
                    </span>
                  </div>
                  {i < 2 && <ChevronRight size={10} className="text-slate-600 flex-shrink-0" />}
                </React.Fragment>
              ))}
            </div>

            {/* Auth Mode Tabs (Sign In vs Sign Up) */}
            {step === 'login' && (
              <div className="flex bg-slate-950/80 p-1 rounded-xl border border-indigo-500/20 mb-6">
                <button
                  type="button"
                  onClick={() => { setAuthMode('signin'); setError(null); }}
                  className={`flex-1 py-2 rounded-lg text-xs font-mono font-bold transition-all flex items-center justify-center gap-1.5 ${
                    authMode === 'signin'
                      ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-lg'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Lock size={12} /> Sign In
                </button>
                <button
                  type="button"
                  onClick={() => { setAuthMode('signup'); setError(null); }}
                  className={`flex-1 py-2 rounded-lg text-xs font-mono font-bold transition-all flex items-center justify-center gap-1.5 ${
                    authMode === 'signup'
                      ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-lg'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <UserPlus size={12} /> Sign Up
                </button>
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="mb-5 px-4 py-3 rounded-xl font-mono text-xs text-red-400 border animate-card-enter"
                style={{ background: 'rgba(239,68,68,0.08)', borderColor: 'rgba(239,68,68,0.3)' }}>
                ⚠ {error}
              </div>
            )}

            {/* ── SIGN IN FORM ── */}
            {step === 'login' && authMode === 'signin' && (
              <form onSubmit={handleLogin} className="space-y-4">
                <div className="space-y-1.5">
                  <label className="font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    Email Address
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    required
                    placeholder="admin@authbrain.com"
                    className="neon-input w-full px-4 py-3 rounded-xl text-sm font-mono"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    Password
                  </label>
                  <div className="relative">
                    <input
                      type={showPass ? 'text' : 'password'}
                      value={password}
                      onChange={e => setPassword(e.target.value)}
                      required
                      placeholder="••••••••••"
                      className="neon-input w-full px-4 py-3 pr-11 rounded-xl text-sm font-mono"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPass(!showPass)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-cyan-400 transition-colors"
                    >
                      {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>

                <div className="pt-2">
                  <NeonButton type="submit" loading={loading} fullWidth size="lg" variant="primary">
                    <Zap size={16} />
                    <span>Sign In & Authenticate</span>
                  </NeonButton>
                </div>

                {/* Demo Quick Login Helper */}
                <button
                  type="button"
                  onClick={fillDemoAdmin}
                  className="w-full text-center font-mono text-[11px] text-cyan-400 hover:text-cyan-300 bg-cyan-950/30 border border-cyan-500/20 py-2 rounded-xl transition-all flex items-center justify-center gap-1.5"
                >
                  <UserCheck size={13} /> Fill Demo Credentials (admin@authbrain.com)
                </button>

                <p className="text-center font-mono text-[10px] text-slate-500 pt-1">
                  AES-256 encrypted · Supabase Postgres · Zero-knowledge auth
                </p>
              </form>
            )}

            {/* ── SIGN UP FORM ── */}
            {step === 'login' && authMode === 'signup' && (
              <form onSubmit={handleRegister} className="space-y-3.5">
                <div className="space-y-1">
                  <label className="font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    Full Name
                  </label>
                  <input
                    type="text"
                    value={fullName}
                    onChange={e => setFullName(e.target.value)}
                    required
                    placeholder="John Smith"
                    className="neon-input w-full px-4 py-2.5 rounded-xl text-sm font-mono"
                  />
                </div>

                <div className="space-y-1">
                  <label className="font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    Email Address
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    required
                    placeholder="user@example.com"
                    className="neon-input w-full px-4 py-2.5 rounded-xl text-sm font-mono"
                  />
                </div>

                <div className="space-y-1">
                  <label className="font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    Password
                  </label>
                  <input
                    type={showPass ? 'text' : 'password'}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                    placeholder="At least 6 characters"
                    className="neon-input w-full px-4 py-2.5 rounded-xl text-sm font-mono"
                  />
                </div>

                <div className="space-y-1">
                  <label className="font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    Confirm Password
                  </label>
                  <input
                    type={showPass ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={e => setConfirmPassword(e.target.value)}
                    required
                    placeholder="Repeat password"
                    className="neon-input w-full px-4 py-2.5 rounded-xl text-sm font-mono"
                  />
                </div>

                <div className="pt-2">
                  <NeonButton type="submit" loading={loading} fullWidth size="lg" variant="primary">
                    <UserPlus size={16} />
                    <span>Create Account & Continue</span>
                  </NeonButton>
                </div>
              </form>
            )}

            {/* ── ENROLLMENT WIZARD ── */}
            {step === 'enrollment' && (
              <FaceEnrollmentWizard
                accessToken={accessToken}
                onComplete={() => setStep('consent')}
              />
            )}

            {/* ── CONSENT FORM ── */}
            {step === 'consent' && (
              <div className="space-y-5">
                <div className="rounded-xl p-4 space-y-3 text-[11px]"
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(79,70,229,0.15)' }}>
                  <h3 className="font-mono text-[10px] uppercase tracking-widest text-violet-400 mb-3">
                    Biometric Data Processing Agreement
                  </h3>
                  {[
                    { icon: '🔬', text: 'Real-time facial landmark & emotion analysis via computer vision' },
                    { icon: '🧠', text: 'Deep learning inference (GNN, transformer ensembles) on facial data' },
                    { icon: '🔐', text: 'Session data encrypted at rest and in transit (AES-256 / TLS 1.3)' },
                    { icon: '⏱', text: 'Data retained only for your session — auto-purged on logout' },
                    { icon: '🛑', text: 'You may withdraw consent at any time by ending the session' },
                  ].map((item, i) => (
                    <div key={i} className="flex items-start gap-2.5 text-slate-300">
                      <span className="text-sm mt-0.5">{item.icon}</span>
                      <span>{item.text}</span>
                    </div>
                  ))}
                </div>

                <label className="flex items-start gap-3 cursor-pointer group">
                  <div className="relative mt-0.5 flex-shrink-0">
                    <input
                      type="checkbox"
                      checked={consentChecked}
                      onChange={e => setConsentChecked(e.target.checked)}
                      className="sr-only"
                    />
                    <div className="w-5 h-5 rounded-md flex items-center justify-center transition-all duration-200"
                      style={{
                        background: consentChecked ? 'linear-gradient(135deg,#4f46e5,#7c3aed)' : 'rgba(10,16,32,0.8)',
                        border: `1px solid ${consentChecked ? '#7c3aed' : 'rgba(79,70,229,0.3)'}`,
                        boxShadow: consentChecked ? '0 0 12px rgba(139,92,246,0.4)' : 'none',
                      }}>
                      {consentChecked && <CheckCircle size={13} className="text-white" />}
                    </div>
                  </div>
                  <span className="text-xs text-slate-300 leading-relaxed group-hover:text-white transition-colors">
                    I understand and freely consent to biometric analysis of my facial data for the
                    purposes described above. I confirm I am aged 18 or over.
                  </span>
                </label>

                <NeonButton
                  onClick={handleConsent}
                  disabled={!consentChecked}
                  loading={loading}
                  fullWidth
                  size="lg"
                  variant="primary"
                >
                  <Shield size={16} />
                  <span>Grant Access & Begin Session</span>
                </NeonButton>

                <button
                  onClick={() => setStep('login')}
                  className="w-full text-center font-mono text-[10px] text-slate-500 hover:text-cyan-400 transition-colors py-1"
                >
                  ← Back to authentication
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom branding */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-10 font-mono text-[9px] text-slate-600 tracking-widest text-center">
        AUTHFACEGRAPH AI PLATFORM · SECURE BIOMETRIC INTELLIGENCE · {new Date().getFullYear()}
      </div>
    </div>
  );
};
