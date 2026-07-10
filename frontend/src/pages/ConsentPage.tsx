/**
 * AuthBrain AI Face Analysis Engine
 * Consent + Login Page
 *
 * Presents the login form and explicit consent workflow.
 * Users must log in AND grant consent before analysis begins.
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, Lock, Eye, EyeOff, Shield, CheckCircle } from 'lucide-react';
import axios from 'axios';
import { useAuthStore, useAnalysisStore } from '../store';
import type { LoginResponse, ConsentResponse } from '../types/analysis';
import clsx from 'clsx';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const ConsentPage: React.FC = () => {
  const navigate    = useNavigate();
  const { setAuth } = useAuthStore();
  const { setConsentGranted, setSessionId, addLog } = useAnalysisStore();

  const [step, setStep]   = useState<'login' | 'consent'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [orgId, setOrgId]  = useState('');
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [consentChecked, setConsentChecked] = useState(false);
  const [accessToken, setAccessToken]       = useState('');

  // ── Step 1: Login ────────────────────────────────────────────────────────

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
      setStep('consent');
      addLog('info', `Logged in as ${res.data.email}`, 'auth');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed. Check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  // ── Step 2: Consent ──────────────────────────────────────────────────────

  const handleConsent = async () => {
    if (!consentChecked) return;
    setLoading(true);
    setError(null);
    try {
      const sessionId = crypto.randomUUID();
      const res = await axios.post<ConsentResponse>(
        `${API_BASE}/api/consent/`,
        {
          session_id: sessionId,
          consent_granted: true,
          consent_text_version: '1.0',
          user_agent: navigator.userAgent,
        },
        { headers: { Authorization: `Bearer ${accessToken}` } }
      );
      setSessionId(res.data.session_id);
      setConsentGranted(true);
      addLog('info', 'Consent recorded. Starting analysis session.', 'consent');
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Consent submission failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-950 flex items-center justify-center p-4"
         style={{ background: 'radial-gradient(ellipse at 50% 30%, rgba(0,255,65,0.06) 0%, #020408 60%)' }}>

      {/* Card */}
      <div className="w-full max-w-md">

        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <h1 className="text-3xl font-extrabold text-white tracking-tight uppercase">AuthFaceGraph</h1>
          <p className="text-dark-400 text-sm font-mono mt-1">Face Analysis Engine</p>
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-2 mb-6 justify-center">
          {['login', 'consent'].map((s, i) => (
            <React.Fragment key={s}>
              <div className={clsx(
                'w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold font-mono border transition-all',
                step === s ? 'bg-brand-500 border-brand-500 text-dark-950' :
                i === 0 && step === 'consent' ? 'bg-brand-500/20 border-brand-500/40 text-brand-500' :
                'bg-dark-800 border-dark-600 text-dark-500'
              )}>
                {i === 0 && step === 'consent' ? <CheckCircle size={12} /> : i + 1}
              </div>
              {i < 1 && <div className={clsx('h-px w-12', step === 'consent' ? 'bg-brand-500/40' : 'bg-dark-700')} />}
            </React.Fragment>
          ))}
        </div>

        {/* Login Form */}
        {step === 'login' && (
          <form onSubmit={handleLogin} className="bg-dark-800/60 border border-dark-600/50 rounded-2xl p-6 space-y-4 backdrop-blur-sm">
            <h2 className="text-lg font-semibold text-white mb-2">Sign in to your account</h2>

            <div>
              <label className="block text-xs font-mono text-dark-400 mb-1.5">Organization ID (optional)</label>
              <input
                type="text"
                value={orgId}
                onChange={e => setOrgId(e.target.value)}
                placeholder="org-slug or leave blank"
                className="w-full bg-dark-900 border border-dark-600 rounded-lg px-3 py-2.5 text-sm text-white
                           placeholder:text-dark-600 focus:outline-none focus:border-brand-500/60 transition-colors font-mono"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-dark-400 mb-1.5">Email</label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                placeholder="your@email.com"
                className="w-full bg-dark-900 border border-dark-600 rounded-lg px-3 py-2.5 text-sm text-white
                           placeholder:text-dark-600 focus:outline-none focus:border-brand-500/60 transition-colors"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-dark-400 mb-1.5">Password</label>
              <div className="relative">
                <input
                  type={showPass ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  placeholder="••••••••"
                  className="w-full bg-dark-900 border border-dark-600 rounded-lg px-3 py-2.5 pr-10 text-sm text-white
                             placeholder:text-dark-600 focus:outline-none focus:border-brand-500/60 transition-colors"
                />
                <button type="button" onClick={() => setShowPass(!showPass)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-dark-500 hover:text-dark-300">
                  {showPass ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>

            {error && (
              <p className="text-risk-critical text-xs font-mono bg-risk-critical/10 border border-risk-critical/30 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-brand-500 hover:bg-brand-400 disabled:opacity-50 text-dark-950 font-bold py-2.5 rounded-lg
                         transition-all shadow-[0_0_20px_rgba(0,255,65,0.3)] hover:shadow-[0_0_30px_rgba(0,255,65,0.5)]"
            >
              {loading ? 'Signing in…' : 'Sign In →'}
            </button>
          </form>
        )}

        {/* Consent Form */}
        {step === 'consent' && (
          <div className="bg-dark-800/60 border border-dark-600/50 rounded-2xl p-6 space-y-5 backdrop-blur-sm">
            <div className="flex items-center gap-3">
              <Shield size={20} className="text-brand-500 flex-shrink-0" />
              <h2 className="text-lg font-semibold text-white">Camera & AI Analysis Consent</h2>
            </div>

            <div className="bg-dark-900/60 border border-dark-700/50 rounded-xl p-4 space-y-3 text-xs font-mono text-dark-300 leading-relaxed">
              <p>By proceeding, you consent to:</p>
              <ul className="space-y-1.5 list-none">
                {[
                  'Your webcam being accessed for real-time facial analysis',
                  'Facial geometry metrics being extracted (not biometric data)',
                  'Session summaries being stored (not video or images)',
                  'AI analysis of attention, fatigue, and behavioral patterns',
                ].map((item, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <CheckCircle size={12} className="text-brand-500 mt-0.5 flex-shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
              <p className="text-dark-500 text-[10px]">
                This is NOT facial recognition. No biometric identifiers are stored.
                You may revoke consent at any time.
              </p>
            </div>

            <label className="flex items-start gap-3 cursor-pointer group">
              <input
                type="checkbox"
                checked={consentChecked}
                onChange={e => setConsentChecked(e.target.checked)}
                className="mt-0.5 accent-brand-500 w-4 h-4"
              />
              <span className="text-sm text-dark-300 group-hover:text-white transition-colors">
                I have read and explicitly consent to the above analysis conditions.
              </span>
            </label>

            {error && (
              <p className="text-risk-critical text-xs font-mono bg-risk-critical/10 border border-risk-critical/30 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <div className="flex gap-3">
              <button onClick={() => setStep('login')}
                      className="flex-1 border border-dark-600 text-dark-400 hover:text-white py-2.5 rounded-lg text-sm transition-colors">
                Back
              </button>
              <button
                onClick={handleConsent}
                disabled={!consentChecked || loading}
                className="flex-1 bg-brand-500 hover:bg-brand-400 disabled:opacity-40 text-dark-950 font-bold py-2.5 rounded-lg
                           text-sm transition-all shadow-[0_0_20px_rgba(0,255,65,0.3)]"
              >
                {loading ? 'Recording…' : 'Consent & Start'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
