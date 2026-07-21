/**
 * AuthFaceGraph — Shared UI Components
 * GlassCard, NeonButton, MetricBadge, PageTransition, StatusDot
 */

import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';

// ════════════════════════════════════════════════════════════════════
// GlassCard
// ════════════════════════════════════════════════════════════════════
interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  glow?: 'cyan' | 'violet' | 'blue' | 'none';
  gradient?: boolean;
  onClick?: () => void;
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className,
  hover = false,
  glow = 'none',
  gradient = false,
  onClick,
}) => {
  const glowClasses = {
    cyan:   'glow-cyan',
    violet: 'glow-violet',
    blue:   'glow-blue',
    none:   '',
  };

  return (
    <div
      onClick={onClick}
      className={clsx(
        'glass rounded-2xl',
        hover && 'glass-hover cursor-pointer transition-all duration-250',
        gradient && 'gradient-border',
        glow !== 'none' && glowClasses[glow],
        onClick && 'cursor-pointer',
        className,
      )}
    >
      {children}
    </div>
  );
};

// ════════════════════════════════════════════════════════════════════
// NeonButton
// ════════════════════════════════════════════════════════════════════
interface NeonButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  loading?: boolean;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  type?: 'button' | 'submit' | 'reset';
  fullWidth?: boolean;
}

export const NeonButton: React.FC<NeonButtonProps> = ({
  children,
  onClick,
  disabled,
  loading,
  variant = 'primary',
  size = 'md',
  className,
  type = 'button',
  fullWidth,
}) => {
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-xs rounded-lg',
    md: 'px-5 py-2.5 text-sm rounded-xl',
    lg: 'px-8 py-3.5 text-base rounded-xl',
  };

  const variantStyles: Record<string, React.CSSProperties> = {
    primary: {
      background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
      color: 'white',
      border: 'none',
    },
    secondary: {
      background: 'rgba(10,16,32,0.6)',
      color: '#c9d4f0',
      border: '1px solid rgba(79,70,229,0.4)',
    },
    ghost: {
      background: 'transparent',
      color: '#c9d4f0',
      border: '1px solid rgba(79,70,229,0.2)',
    },
    danger: {
      background: 'linear-gradient(135deg, #991b1b, #dc2626)',
      color: 'white',
      border: 'none',
    },
  };

  const hoverGlows: Record<string, string> = {
    primary: 'hover:shadow-[0_0_20px_rgba(139,92,246,0.4)]',
    secondary: 'hover:border-[rgba(0,212,255,0.5)] hover:shadow-[0_0_16px_rgba(0,212,255,0.2)]',
    ghost: 'hover:bg-[rgba(79,70,229,0.1)]',
    danger: 'hover:shadow-[0_0_20px_rgba(239,68,68,0.4)]',
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      style={variantStyles[variant]}
      className={clsx(
        'font-semibold tracking-wide transition-all duration-200 ease-out',
        'flex items-center justify-center gap-2',
        'disabled:opacity-40 disabled:cursor-not-allowed',
        'hover:-translate-y-[1px] active:translate-y-0',
        sizeClasses[size],
        hoverGlows[variant],
        fullWidth && 'w-full',
        className,
      )}
    >
      {loading ? (
        <span className="flex items-center gap-2">
          <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Processing...
        </span>
      ) : children}
    </button>
  );
};

// ════════════════════════════════════════════════════════════════════
// MetricBadge
// ════════════════════════════════════════════════════════════════════
interface MetricBadgeProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  color?: 'cyan' | 'violet' | 'blue' | 'green' | 'amber' | 'red';
  animated?: boolean;
  className?: string;
}

export const MetricBadge: React.FC<MetricBadgeProps> = ({
  label,
  value,
  icon,
  color = 'cyan',
  animated = false,
  className,
}) => {
  const colorMap = {
    cyan:   { text: '#00d4ff', bg: 'rgba(0,212,255,0.08)', border: 'rgba(0,212,255,0.2)' },
    violet: { text: '#8b5cf6', bg: 'rgba(139,92,246,0.08)', border: 'rgba(139,92,246,0.2)' },
    blue:   { text: '#3b82f6', bg: 'rgba(59,130,246,0.08)', border: 'rgba(59,130,246,0.2)' },
    green:  { text: '#10b981', bg: 'rgba(16,185,129,0.08)', border: 'rgba(16,185,129,0.2)' },
    amber:  { text: '#f59e0b', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.2)' },
    red:    { text: '#ef4444', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.2)' },
  };
  const c = colorMap[color];

  return (
    <div
      className={clsx('flex items-center gap-2 px-3 py-1.5 rounded-xl', animated && 'animate-count-up', className)}
      style={{ background: c.bg, border: `1px solid ${c.border}` }}
    >
      {icon && <span style={{ color: c.text }}>{icon}</span>}
      <div className="font-mono text-[10px]">
        <span className="block text-[8px] uppercase tracking-widest opacity-60" style={{ color: c.text }}>
          {label}
        </span>
        <span className="block font-bold" style={{ color: c.text }}>
          {value}
        </span>
      </div>
    </div>
  );
};

// ════════════════════════════════════════════════════════════════════
// PageTransition
// ════════════════════════════════════════════════════════════════════
interface PageTransitionProps {
  children: React.ReactNode;
  id?: string;
}

export const PageTransition: React.FC<PageTransitionProps> = ({ children, id }) => {
  return (
    <div
      key={id}
      className="section-transition w-full h-full"
      style={{ willChange: 'transform, opacity' }}
    >
      {children}
    </div>
  );
};

// ════════════════════════════════════════════════════════════════════
// StatusDot
// ════════════════════════════════════════════════════════════════════
interface StatusDotProps {
  status: 'online' | 'offline' | 'warning' | 'error';
  label?: string;
  className?: string;
}

export const StatusDot: React.FC<StatusDotProps> = ({ status, label, className }) => {
  const colors = {
    online:  '#10b981',
    offline: '#6b7280',
    warning: '#f59e0b',
    error:   '#ef4444',
  };
  return (
    <div className={clsx('flex items-center gap-1.5', className)}>
      <div
        className="w-1.5 h-1.5 rounded-full animate-pulse"
        style={{ background: colors[status], boxShadow: `0 0 6px ${colors[status]}` }}
      />
      {label && (
        <span className="font-mono text-[10px] uppercase tracking-wide" style={{ color: colors[status] }}>
          {label}
        </span>
      )}
    </div>
  );
};

// ════════════════════════════════════════════════════════════════════
// LoadingSkeleton
// ════════════════════════════════════════════════════════════════════
export const LoadingSkeleton: React.FC<{ className?: string }> = ({ className }) => (
  <div className={clsx('animate-shimmer rounded-lg', className)} style={{ minHeight: 12 }} />
);

// ════════════════════════════════════════════════════════════════════
// SectionHeader
// ════════════════════════════════════════════════════════════════════
interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({ title, subtitle, icon, actions }) => (
  <div className="flex items-center justify-between mb-5">
    <div className="flex items-center gap-3">
      {icon && (
        <div className="w-9 h-9 rounded-xl flex items-center justify-center"
          style={{ background: 'rgba(79,70,229,0.15)', border: '1px solid rgba(79,70,229,0.3)' }}>
          <span className="text-violet-400">{icon}</span>
        </div>
      )}
      <div>
        <h2 className="text-sm font-bold tracking-wide text-white">{title}</h2>
        {subtitle && <p className="text-[11px] text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
    {actions && <div className="flex items-center gap-2">{actions}</div>}
  </div>
);
