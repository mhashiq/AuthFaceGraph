/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // AuthFaceGraph brand palette — neon cyan / violet / indigo
        brand: {
          50:  '#e0faff',
          100: '#b0f0ff',
          200: '#7ae5ff',
          300: '#3dd9ff',
          400: '#00d4ff',  // Neon cyan — primary accent
          500: '#00b8e0',
          600: '#0090b3',
          700: '#006b87',
          800: '#00475a',
          900: '#00232e',
        },
        violet: {
          300: '#c4b5fd',
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
          700: '#6d28d9',
        },
        // Dark backgrounds
        dark: {
          950: '#010409',
          900: '#030712',
          850: '#070d1a',
          800: '#0e1628',
          750: '#111c33',
          700: '#162038',
          600: '#1d2b4a',
          500: '#243460',
          400: '#3a4f70',
          300: '#5d7399',
        },
        // Status colors
        risk: {
          critical: '#ef4444',
          high:     '#f97316',
          medium:   '#f59e0b',
          low:      '#10b981',
        },
      },
      fontFamily: {
        sans:    ['Inter', 'system-ui', 'sans-serif'],
        mono:    ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Orbitron', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow':    'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow':          'glow 2s ease-in-out infinite alternate',
        'scan-line':     'scanLine 4s linear infinite',
        'slide-in-right':'slideInRight 0.3s ease-out',
        'fade-in':       'fadeIn 0.5s ease-out',
        'counter-spin':  'spin 1s linear infinite',
      },
      keyframes: {
        glow: {
          '0%':   { boxShadow: '0 0 5px #00ff41, 0 0 10px #00ff41' },
          '100%': { boxShadow: '0 0 20px #00ff41, 0 0 40px #00ff41, 0 0 80px #00ff41' },
        },
        scanLine: {
          '0%':   { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        slideInRight: {
          '0%':   { transform: 'translateX(100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        fadeIn: {
          '0%':   { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      backgroundImage: {
        'grid-pattern': `linear-gradient(rgba(0,255,65,0.03) 1px, transparent 1px),
                         linear-gradient(90deg, rgba(0,255,65,0.03) 1px, transparent 1px)`,
        'radial-glow':  'radial-gradient(ellipse at center, rgba(0,255,65,0.1) 0%, transparent 70%)',
      },
    },
  },
  plugins: [],
};
