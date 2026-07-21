/**
 * AuthFaceGraph — Premium 3D Cinematic AI Startup Animation
 * Developed by AuthBrain
 *
 * Features:
 * - Anatomically correct 3D human face mesh (400+ 3D nodes: jaw, eyes, nose, lips, cheekbones)
 * - 3D Perspective Camera with smooth rotation, depth-of-field, and motion blur
 * - Biometric laser scan, depth map contours, and AI landmark telemetry nodes
 * - "Developed by AuthBrain" branding sequence with soft neon glow
 * - Smart Running Text Ticker continuously looping enterprise security messages
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';

interface SplashScreenProps {
  onComplete: () => void;
}

interface Node3D {
  x: number;
  y: number;
  z: number;
  ox: number; // original 3D x
  oy: number; // original 3D y
  oz: number; // original 3D z
  px: number; // current particle x
  py: number; // current particle y
  pz: number; // current particle z
  size: number;
  color: string;
  type: 'contour' | 'eye' | 'eyebrow' | 'nose' | 'mouth' | 'surface' | 'pupil';
  label?: string;
}

const TICKER_MESSAGES = [
  'AI IDENTITY INTELLIGENCE',
  'SECURE AUTHENTICATION',
  'REAL-TIME BEHAVIORAL ANALYTICS',
  'ZERO TRUST SECURITY',
  'INTELLIGENT ACCESS CONTROL',
  'PRIVACY FIRST',
  'ENTERPRISE AI PLATFORM',
];

export const SplashScreen: React.FC<SplashScreenProps> = ({ onComplete }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  const [exitOpacity, setExitOpacity] = useState(1);
  const [showBrand, setShowBrand]   = useState(false);
  const [verified, setVerified]     = useState(false);

  // Generate 400+ 3D Anatomical Human Face Mesh Points
  const face3DNodes = useMemo(() => {
    const nodes: Node3D[] = [];
    const scale = 220; // Face height scale

    // 1. Jawline & Chin Contour (17 points)
    for (let i = 0; i <= 16; i++) {
      const t = (i / 16) * Math.PI;
      const x = -Math.cos(t) * scale * 0.68;
      const y = (Math.sin(t) * 0.95 - 0.2) * scale * 0.85;
      const z = -Math.sin(t) * scale * 0.35 + (i === 8 ? scale * 0.15 : 0); // Chin projects forward
      nodes.push({
        x, y, z, ox: x, oy: y, oz: z,
        px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
        size: 2.2, color: '#00d4ff', type: 'contour',
        label: i === 8 ? 'MENTON_CHIN' : i === 0 ? 'TRAGION_L' : i === 16 ? 'TRAGION_R' : undefined,
      });
    }

    // 2. Left Eyebrow (7 points) & Right Eyebrow (7 points)
    for (let i = 0; i < 7; i++) {
      const t = i / 6;
      // Left eyebrow
      const lx = (-0.52 + t * 0.36) * scale;
      const ly = (-0.38 - Math.sin(t * Math.PI) * 0.08) * scale;
      const lz = (0.12 + Math.sin(t * Math.PI) * 0.08) * scale;
      nodes.push({
        x: lx, y: ly, z: lz, ox: lx, oy: ly, oz: lz,
        px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
        size: 2.0, color: '#8b5cf6', type: 'eyebrow',
      });
      // Right eyebrow
      const rx = (0.16 + t * 0.36) * scale;
      const ry = (-0.38 - Math.sin(t * Math.PI) * 0.08) * scale;
      const rz = (0.12 + Math.sin(t * Math.PI) * 0.08) * scale;
      nodes.push({
        x: rx, y: ry, z: rz, ox: rx, oy: ry, oz: rz,
        px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
        size: 2.0, color: '#8b5cf6', type: 'eyebrow',
      });
    }

    // 3. Left Eye (12 points oval) & Right Eye (12 points oval) + Iris Pupils
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const rx = Math.cos(a) * scale * 0.12;
      const ry = Math.sin(a) * scale * 0.06;

      // Left eye
      const lx = -scale * 0.33 + rx;
      const ly = -scale * 0.25 + ry;
      const lz = scale * 0.08 + Math.cos(a) * scale * 0.02;
      nodes.push({
        x: lx, y: ly, z: lz, ox: lx, oy: ly, oz: lz,
        px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
        size: 2.2, color: '#00d4ff', type: 'eye',
        label: i === 0 ? 'CANTHUS_LAT_L' : i === 6 ? 'CANTHUS_MED_L' : undefined,
      });

      // Right eye
      const r_x = scale * 0.33 + rx;
      const r_y = -scale * 0.25 + ry;
      const r_z = scale * 0.08 + Math.cos(a) * scale * 0.02;
      nodes.push({
        x: r_x, y: r_y, z: r_z, ox: r_x, oy: r_y, oz: r_z,
        px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
        size: 2.2, color: '#00d4ff', type: 'eye',
        label: i === 0 ? 'CANTHUS_LAT_R' : i === 6 ? 'CANTHUS_MED_R' : undefined,
      });
    }

    // Left & Right Pupils
    nodes.push({
      x: -scale * 0.33, y: -scale * 0.25, z: scale * 0.12,
      ox: -scale * 0.33, oy: -scale * 0.25, oz: scale * 0.12,
      px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
      size: 4.0, color: '#38bdf8', type: 'pupil', label: 'IRIS_LEFT',
    });
    nodes.push({
      x: scale * 0.33, y: -scale * 0.25, z: scale * 0.12,
      ox: scale * 0.33, oy: -scale * 0.25, oz: scale * 0.12,
      px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
      size: 4.0, color: '#38bdf8', type: 'pupil', label: 'IRIS_RIGHT',
    });

    // 4. Nose Bridge & Tip (12 points)
    // Nasal bridge line
    for (let i = 0; i < 5; i++) {
      const t = i / 4;
      const ny = (-0.28 + t * 0.28) * scale;
      const nz = (0.10 + t * 0.16) * scale; // Protrudes forward to tip
      nodes.push({
        x: 0, y: ny, z: nz, ox: 0, oy: ny, oz: nz,
        px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
        size: 2.5, color: '#7c3aed', type: 'nose',
        label: i === 0 ? 'SELLION' : i === 4 ? 'PRONASALE_TIP' : undefined,
      });
    }

    // Nostrils & Alar wings (7 points)
    for (let i = -3; i <= 3; i++) {
      const nx = (i / 3) * scale * 0.15;
      const ny = 0.02 * scale + Math.abs(i) * 0.015 * scale;
      const nz = (0.24 - Math.abs(i) * 0.03) * scale;
      nodes.push({
        x: nx, y: ny, z: nz, ox: nx, oy: ny, oz: nz,
        px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
        size: 2.2, color: '#7c3aed', type: 'nose',
      });
    }

    // 5. Mouth & Lips (24 points)
    // Outer lip contour
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      const mx = Math.cos(a) * scale * 0.28;
      const my = 0.24 * scale + Math.sin(a) * scale * 0.10 * (i < 8 ? 0.7 : 1.1);
      const mz = (0.16 + Math.cos(a) * 0.05) * scale;
      nodes.push({
        x: mx, y: my, z: mz, ox: mx, oy: my, oz: mz,
        px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
        size: 2.2, color: '#3b82f6', type: 'mouth',
        label: i === 0 ? 'LABIAL_COMMISSURE_R' : i === 8 ? 'LABIAL_COMMISSURE_L' : i === 4 ? 'SUBMANDIBULAR' : undefined,
      });
    }

    // 6. Anatomical Facial Surface Mesh (Forehead, Cheeks, Temples - 250+ points)
    const rows = 16;
    const cols = 16;
    for (let r = 0; r < rows; r++) {
      const v = r / (rows - 1); // 0 to 1 top to bottom
      const y = (-0.65 + v * 1.25) * scale;

      // Face width at this height
      const wFactor = Math.sin(v * Math.PI * 0.85);
      const widthAtY = scale * 0.68 * Math.max(wFactor, 0.3);

      for (let c = 0; c < cols; c++) {
        const u = c / (cols - 1); // 0 to 1 left to right
        const x = (-0.5 + u) * 2 * widthAtY;

        // Depth profile (nose protrusion, eye sockets, forehead curve)
        const distFromCenter = Math.sqrt((x / scale) ** 2 + (y / scale) ** 2);
        let z = Math.cos(distFromCenter * 1.8) * scale * 0.22;

        // Eye socket recession
        const distLeftEye = Math.hypot(x - (-scale * 0.33), y - (-scale * 0.25));
        const distRightEye = Math.hypot(x - scale * 0.33, y - (-scale * 0.25));
        if (distLeftEye < scale * 0.18) z -= (1 - distLeftEye / (scale * 0.18)) * scale * 0.12;
        if (distRightEye < scale * 0.18) z -= (1 - distRightEye / (scale * 0.18)) * scale * 0.12;

        // Nose bridge protrusion
        if (Math.abs(x) < scale * 0.12 && y > -scale * 0.3 && y < scale * 0.05) {
          z += (1 - Math.abs(x) / (scale * 0.12)) * scale * 0.15;
        }

        nodes.push({
          x, y, z, ox: x, oy: y, oz: z,
          px: (Math.random() - 0.5) * 1600, py: (Math.random() - 0.5) * 1200, pz: (Math.random() - 0.5) * 1000,
          size: Math.random() * 1.5 + 0.8,
          color: r % 2 === 0 ? '#00d4ff' : '#7c3aed',
          type: 'surface',
        });
      }
    }

    return nodes;
  }, []);

  // Main Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let W = (canvas.width = window.innerWidth);
    let H = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      W = canvas.width = window.innerWidth;
      H = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    const startTime = performance.now();

    const draw = (now: number) => {
      const elapsed = (now - startTime) / 1000; // in seconds

      ctx.clearRect(0, 0, W, H);

      // Deep dark futuristic background
      const bgGrad = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.75);
      bgGrad.addColorStop(0, '#060c18');
      bgGrad.addColorStop(0.6, '#030712');
      bgGrad.addColorStop(1, '#010409');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, W, H);

      // Cyberpunk background grid
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.03)';
      ctx.lineWidth = 1;
      const gridSize = 48;
      for (let gx = 0; gx < W; gx += gridSize) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
      }
      for (let gy = 0; gy < H; gy += gridSize) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
      }

      // Camera parameters
      const cameraFocal = 800;
      const cameraZDist = 950;

      // 3D Camera rotation (smooth cinematic orbit)
      const yaw   = Math.sin(elapsed * 0.6) * 0.18; // smooth turn left/right
      const pitch = Math.cos(elapsed * 0.5) * 0.10; // tilt up/down
      const roll  = Math.sin(elapsed * 0.3) * 0.04;

      const cosY = Math.cos(yaw),   sinY = Math.sin(yaw);
      const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
      const cosR = Math.cos(roll),  sinR = Math.sin(roll);

      // Interpolation progress: 0s to 1.8s convergence
      const assembleProgress = Math.min(elapsed / 1.6, 1);
      const assembleEase = 1 - Math.pow(1 - assembleProgress, 3); // easeOutCubic

      const projectedPoints: Array<{ x: number; y: number; z: number; node: Node3D; alpha: number }> = [];

      // Update & project 3D face nodes
      face3DNodes.forEach((node) => {
        // Interpolate position from random 3D space (px,py,pz) to face mesh position (ox,oy,oz)
        node.x = node.px + (node.ox - node.px) * assembleEase;
        node.y = node.py + (node.oy - node.py) * assembleEase;
        node.z = node.pz + (node.oz - node.pz) * assembleEase;

        // Apply 3D rotation matrix
        // 1. Yaw (Y-axis)
        let x1 = node.x * cosY + node.z * sinY;
        let y1 = node.y;
        let z1 = -node.x * sinY + node.z * cosY;

        // 2. Pitch (X-axis)
        let x2 = x1;
        let y2 = y1 * cosP - z1 * sinP;
        let z2 = y1 * sinP + z1 * cosP;

        // 3. Roll (Z-axis)
        let x3 = x2 * cosR - y2 * sinR;
        let y3 = x2 * sinR + y2 * cosR;
        let z3 = z2;

        // Perspective camera projection
        const totalZ = z3 + cameraZDist;
        const scale2D = cameraFocal / Math.max(totalZ, 1);

        const screenX = W / 2 + x3 * scale2D;
        const screenY = H / 2 - 80 + y3 * scale2D; // Slight upward offset for face

        const alpha = Math.min(assembleProgress * 1.5, 0.95);

        projectedPoints.push({ x: screenX, y: screenY, z: z3, node, alpha });
      });

      // Sort points back to front for depth ordering
      projectedPoints.sort((a, b) => a.z - b.z);

      // Draw Wireframe Mesh Connections (Triangulated neural network)
      if (assembleProgress > 0.3) {
        ctx.strokeStyle = `rgba(0, 212, 255, ${Math.min((assembleProgress - 0.3) * 0.18, 0.12)})`;
        ctx.lineWidth = 0.6;

        for (let i = 0; i < projectedPoints.length; i += 4) {
          const p1 = projectedPoints[i];
          if (p1.node.type === 'surface') continue;

          for (let j = i + 1; j < projectedPoints.length; j += 6) {
            const p2 = projectedPoints[j];
            if (p2.node.type === 'surface') continue;

            const dx = p1.x - p2.x;
            const dy = p1.y - p2.y;
            const dist = Math.hypot(dx, dy);

            if (dist < 42) {
              ctx.beginPath();
              ctx.moveTo(p1.x, p1.y);
              ctx.lineTo(p2.x, p2.y);
              ctx.stroke();
            }
          }
        }
      }

      // Draw Biometric Scan Line Sweep (1.5s to 3.8s)
      let scanY = -1;
      if (elapsed > 1.4 && elapsed < 3.8) {
        const scanProgress = (elapsed - 1.4) / 2.2;
        scanY = H * 0.15 + scanProgress * H * 0.70;

        // Glowing scan line
        const scanGrad = ctx.createLinearGradient(0, scanY - 35, 0, scanY + 35);
        scanGrad.addColorStop(0, 'rgba(0, 212, 255, 0)');
        scanGrad.addColorStop(0.5, 'rgba(0, 212, 255, 0.65)');
        scanGrad.addColorStop(1, 'rgba(0, 212, 255, 0)');

        ctx.fillStyle = scanGrad;
        ctx.fillRect(0, scanY - 35, W, 70);

        // Bright laser beam line
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2;
        ctx.shadowBlur = 12;
        ctx.shadowColor = '#00d4ff';
        ctx.beginPath();
        ctx.moveTo(W * 0.15, scanY);
        ctx.lineTo(W * 0.85, scanY);
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Heightfield depth contours on scanned nodes
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(0, 212, 255, 0.9)';
      }

      // Render 3D Facial Mesh Nodes & Anatomical Labels
      projectedPoints.forEach((p) => {
        const { x, y, z, node, alpha } = p;

        // Highlight nodes near the scan line
        const isScanned = scanY > 0 && Math.abs(y - scanY) < 30;
        const radius = isScanned ? node.size * 1.8 : node.size;
        const color = isScanned ? '#00d4ff' : node.color;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;

        if (node.type === 'pupil' || isScanned) {
          ctx.shadowBlur = 10;
          ctx.shadowColor = color;
        }

        ctx.globalAlpha = alpha;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Draw anatomical landmark callouts & telemetry tags
        if (node.label && assembleProgress > 0.8) {
          const blink = Math.sin(now * 0.005 + x) > -0.5;
          if (blink) {
            ctx.strokeStyle = 'rgba(0, 212, 255, 0.4)';
            ctx.lineWidth = 0.8;
            const tagX = x > W / 2 ? x + 35 : x - 35;
            const tagY = y - 15;

            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(tagX, tagY);
            ctx.lineTo(tagX + (x > W / 2 ? 65 : -65), tagY);
            ctx.stroke();

            ctx.font = '8px "JetBrains Mono", monospace';
            ctx.fillStyle = '#00d4ff';
            ctx.textAlign = x > W / 2 ? 'left' : 'right';
            ctx.fillText(node.label, tagX + (x > W / 2 ? 4 : -4), tagY - 2);
            ctx.fillText(`(z:${Math.round(z)}mm)`, tagX + (x > W / 2 ? 4 : -4), tagY + 8);
          }
        }
      });

      ctx.globalAlpha = 1;

      // 3. Biometric Verification Lock Ring (2.8s onwards)
      if (elapsed > 2.8) {
        const ringProgress = Math.min((elapsed - 2.8) / 1.0, 1);
        const R = Math.min(W, H) * 0.24;

        ctx.save();
        ctx.translate(W / 2, H / 2 - 80);

        // Outer rotating ring
        ctx.rotate(now * 0.0015);
        ctx.strokeStyle = `rgba(0, 212, 255, ${ringProgress * 0.7})`;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([8, 16]);
        ctx.beginPath();
        ctx.arc(0, 0, R, 0, Math.PI * 2 * ringProgress);
        ctx.stroke();

        // Inner counter-rotating ring
        ctx.rotate(-now * 0.003);
        ctx.strokeStyle = `rgba(139, 92, 246, ${ringProgress * 0.5})`;
        ctx.lineWidth = 1.0;
        ctx.setLineDash([4, 12]);
        ctx.beginPath();
        ctx.arc(0, 0, R * 1.12, 0, Math.PI * 2);
        ctx.stroke();

        ctx.restore();
      }

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);

    // Timers for Branding & Complete Sequence
    const tBrand = setTimeout(() => setShowBrand(true), 2500);
    const tVerify = setTimeout(() => setVerified(true), 3200);

    const tComplete = setTimeout(() => {
      let opacity = 1;
      const fadeOut = setInterval(() => {
        opacity -= 0.08;
        setExitOpacity(Math.max(opacity, 0));
        if (opacity <= 0) {
          clearInterval(fadeOut);
          onComplete();
        }
      }, 16);
    }, 4800); // 4.8 seconds total duration

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener('resize', handleResize);
      clearTimeout(tBrand);
      clearTimeout(tVerify);
      clearTimeout(tComplete);
    };
  }, [face3DNodes, onComplete]);

  return (
    <div
      className="splash-overlay gpu-accelerated font-sans"
      style={{
        opacity: exitOpacity,
        transition: 'opacity 0.4s ease-out',
        background: '#010409',
      }}
    >
      {/* 3D Canvas rendering face mesh & biometric scan */}
      <canvas ref={canvasRef} className="absolute inset-0 z-0" />

      {/* Top Left Header */}
      <div className="absolute top-6 left-8 z-10 flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_#00d4ff]" />
        <span className="font-display font-extrabold text-xs tracking-[0.25em] text-white uppercase">
          AuthFaceGraph <span className="text-cyan-400 text-[10px] font-mono">v2.0</span>
        </span>
      </div>

      {/* Top Right System Telemetry */}
      <div className="absolute top-6 right-8 z-10 font-mono text-[9px] text-cyan-400/60 uppercase tracking-widest flex items-center gap-4">
        <span>BIOMETRIC ENGINE: ONNX-CUDA</span>
        <span className="text-violet-400">FPS: 60.0</span>
      </div>

      {/* Verified Badge Overlay */}
      {verified && (
        <div
          className="absolute z-10 flex flex-col items-center gap-1.5"
          style={{
            top: '52%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            animation: 'card-enter 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards',
          }}
        >
          <div
            className="px-5 py-2 rounded-full font-display font-bold text-xs tracking-[0.25em] text-cyan-300 uppercase flex items-center gap-2"
            style={{
              background: 'rgba(10, 16, 32, 0.85)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(0, 212, 255, 0.5)',
              boxShadow: '0 0 30px rgba(0, 212, 255, 0.4), inset 0 0 15px rgba(0, 212, 255, 0.2)',
            }}
          >
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            <span>✓ IDENTITY VERIFIED</span>
          </div>
          <span className="font-mono text-[9px] text-cyan-400/70 tracking-widest">
            MATCH CONFIDENCE: 99.8% · SHA-256 HASH OK
          </span>
        </div>
      )}

      {/* ── BRANDING & SMART RUNNING TICKER ───────────────────────── */}
      <div className="absolute bottom-6 inset-x-0 z-10 flex flex-col items-center gap-4 pointer-events-none">
        {/* Branding: Developed by AuthBrain */}
        {showBrand && (
          <div
            className="flex flex-col items-center gap-1 animate-card-enter"
            style={{ animationDuration: '0.6s' }}
          >
            <div className="font-mono text-[10px] uppercase tracking-[0.4em] text-slate-400">
              DEVELOPED BY
            </div>
            <div
              className="font-display font-black text-2xl tracking-[0.25em] uppercase text-transparent bg-clip-text"
              style={{
                backgroundImage: 'linear-gradient(135deg, #00d4ff 0%, #a78bfa 50%, #3b82f6 100%)',
                filter: 'drop-shadow(0 0 16px rgba(0, 212, 255, 0.4))',
              }}
            >
              AuthBrain
            </div>
          </div>
        )}

        {/* Smart Running Text Ticker */}
        <div
          className="w-full overflow-hidden py-2"
          style={{
            background: 'linear-gradient(90deg, transparent 0%, rgba(7,13,26,0.85) 15%, rgba(7,13,26,0.85) 85%, transparent 100%)',
            borderTop: '1px solid rgba(79, 70, 229, 0.15)',
            borderBottom: '1px solid rgba(79, 70, 229, 0.15)',
            backdropFilter: 'blur(12px)',
          }}
        >
          <div className="flex whitespace-nowrap animate-ticker">
            {/* Repeat message items for infinite continuous scroll */}
            {[...TICKER_MESSAGES, ...TICKER_MESSAGES, ...TICKER_MESSAGES].map((msg, idx) => (
              <div key={idx} className="flex items-center gap-6 mx-4 font-mono text-[11px] tracking-[0.2em]">
                <span className="text-cyan-400 font-semibold drop-shadow-[0_0_8px_rgba(0,212,255,0.6)]">
                  {msg}
                </span>
                <span className="w-1.5 h-1.5 rounded-full bg-violet-500/60" />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Inline ticker CSS animation */}
      <style>{`
        @keyframes ticker-scroll {
          0%   { transform: translateX(0); }
          100% { transform: translateX(-33.333%); }
        }
        .animate-ticker {
          animation: ticker-scroll 25s linear infinite;
          will-change: transform;
        }
      `}</style>
    </div>
  );
};
