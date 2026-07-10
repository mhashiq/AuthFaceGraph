/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Primary brand palette
        brand: {
          50:  '#eeffee',
          100: '#ccffcc',
          200: '#99ff99',
          300: '#66ff66',
          400: '#33ff33',
          500: '#00ff41',  // Matrix green — primary accent
          600: '#00cc34',
          700: '#009927',
          800: '#00661a',
          900: '#00330d',
        },
        // Dark backgrounds
        dark: {
          950: '#020408',
          900: '#050a0e',
          850: '#080f14',
          800: '#0d1520',
          750: '#111c28',
          700: '#162030',
          600: '#1e2d40',
          500: '#263550',
        },
        // Status colors
        risk: {
          critical: '#ff2d55',
          high:     '#ff6b35',
          medium:   '#ffd60a',
          low:      '#34c759',
        },
        // Metric indicators
        metric: {
          ear:     '#00d4ff',
          pose:    '#bf5af2',
          smile:   '#ff9f0a',
          blink:   '#30d158',
          fatigue: '#ff453a',
          focus:   '#64d2ff',
        },
      },
      fontFamily: {
        sans:  ['Inter', 'system-ui', 'sans-serif'],
        mono:  ['JetBrains Mono', 'Fira Code', 'monospace'],
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
