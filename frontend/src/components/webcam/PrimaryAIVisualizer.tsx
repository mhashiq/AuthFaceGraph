import React, { useRef, useEffect, useState } from 'react';
import { Camera, CameraOff, Rotate3d, Maximize2, Minimize2, Eye, EyeOff, Layout } from 'lucide-react';
import { useAnalysisStore } from '../../store';
import { useFaceAnalysis } from '../../hooks/useFaceAnalysis';
import type { Landmark } from '../../types/analysis';

// Local replica of landmark indices for clean typescript access
const LANDMARK_REGIONS = {
  rightEye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
  leftEye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  leftEyebrow: [70, 63, 105, 66, 107, 55, 117, 124],
  rightEyebrow: [300, 293, 334, 296, 336, 285, 346, 353],
  nose: [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 323, 356, 93, 132, 324, 141],
  lips: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 405, 321, 321, 375, 375],
};

const getRegionName = (idx: number): string => {
  if (LANDMARK_REGIONS.leftEye.includes(idx)) return 'Left Eye';
  if (LANDMARK_REGIONS.rightEye.includes(idx)) return 'Right Eye';
  if (LANDMARK_REGIONS.leftEyebrow.includes(idx)) return 'Left Eyebrow';
  if (LANDMARK_REGIONS.rightEyebrow.includes(idx)) return 'Right Eyebrow';
  if (LANDMARK_REGIONS.nose.includes(idx)) return 'Nose';
  if (LANDMARK_REGIONS.lips.includes(idx)) return 'Lips';
  return 'Facial Contour';
};

const MESH_CONNECTIONS = [
  // Face outline / oval
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356],
  [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379],
  [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149],
  [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234],
  [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
  // Left Eye
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133],
  [133, 173], [173, 157], [157, 158], [158, 159], [159, 160], [160, 161], [161, 246], [246, 33],
  // Right Eye
  [362, 382], [382, 381], [381, 380], [380, 374], [374, 373], [373, 390], [390, 249],
  [249, 263], [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384],
  [384, 398], [398, 362],
  // Left Eyebrow
  [70, 63], [63, 105], [105, 66], [66, 107], [107, 55], [55, 117], [117, 124], [124, 70],
  // Right Eyebrow
  [300, 293], [293, 334], [334, 296], [296, 336], [336, 285], [285, 346], [346, 353], [353, 300],
  // Nose Bridge & Base
  [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19], [19, 94], [94, 2],
  [2, 323], [323, 356], [2, 93], [93, 132], [94, 324], [324, 323], [94, 141], [141, 93],
  // Lips / Mouth
  [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270],
  [270, 409], [409, 291], [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314],
  [314, 405], [405, 321], [321, 375], [375, 291]
];

export const PrimaryAIVisualizer: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { videoRef, isRunning, cameraError, start, stop } = useFaceAnalysis();
  const latestResult = useAnalysisStore(s => s.latestResult);
  const activeAlerts = useAnalysisStore(s => s.activeAlerts);
  const [isResearchMode, setResearchMode] = useState(false);

  // Toggles for different viz options
  const [showCamera, setShowCamera] = useState(true);
  const [showMesh, setShowMesh] = useState(true);
  const [showGraph, setShowGraph] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [showXAI, setShowXAI] = useState(true);

  // 3D rotation and zoom variables
  const [yaw, setYaw] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [zoom, setZoom] = useState(1.0);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  // Node Hover Tooltip
  const [hoveredNode, setHoveredNode] = useState<{
    id: number;
    region: string;
    importance: number;
    x: number;
    y: number;
    z: number;
    screenX: number;
    screenY: number;
  } | null>(null);

  // Keep track of projected coordinates for hover calculations
  const projectedPointsRef = useRef<{ x: number; y: number; idx: number }[]>([]);

  // Particle flows animation state
  const particleOffsetRef = useRef(0);

  useEffect(() => {
    let animationFrameId: number;

    const draw = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Update particle flows
      particleOffsetRef.current = (particleOffsetRef.current + 0.02) % 1.0;

      // Set canvas dimension dynamically to match display width
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
      }

      const width = rect.width;
      const height = rect.height;

      // 1. Draw camera video background
      ctx.fillStyle = '#020408';
      ctx.fillRect(0, 0, width, height);

      if (showCamera && isRunning && video && video.readyState >= 2) {
        ctx.save();
        // Set context alpha and apply grayscale matrix for premium scientific overlay feel
        ctx.globalAlpha = 0.45;
        ctx.filter = 'grayscale(60%) contrast(120%)';
        
        // Render video background maintaining aspect ratio (cover)
        const vWidth = video.videoWidth;
        const vHeight = video.videoHeight;
        const videoRatio = vWidth / vHeight;
        const canvasRatio = width / height;

        let drawW = width;
        let drawH = height;
        let dx = 0;
        let dy = 0;

        if (canvasRatio > videoRatio) {
          drawH = width / videoRatio;
          dy = (height - drawH) / 2;
        } else {
          drawW = height * videoRatio;
          dx = (width - drawW) / 2;
        }

        ctx.drawImage(video, dx, dy, drawW, drawH);
        ctx.restore();
      }

      // Draw Grid Overlay (Scientific HUD vibe)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 30; x < width; x += 40) {
        ctx.moveTo(x, 0); ctx.lineTo(x, height);
      }
      for (let y = 30; y < height; y += 40) {
        ctx.moveTo(0, y); ctx.lineTo(width, y);
      }
      ctx.stroke();

      const dl = latestResult?.deep_learning;
      const lms = dl?.landmarks || [];

      if (isRunning && lms.length > 0) {
        // Determine if we should use orbit projection or direct mapping
        const useOrbit = !showCamera || yaw !== 0 || pitch !== 0 || zoom !== 1.0;
        const projected: { x: number; y: number; z: number; importance: number; idx: number }[] = [];

        if (useOrbit) {
          // Centered 3D Auto-scaled view (with Yaw/Pitch/Zoom rotation)
          let minX = 1, maxX = 0, minY = 1, maxY = 0;
          lms.forEach((lm: Landmark) => {
            if (lm.x < minX) minX = lm.x;
            if (lm.x > maxX) maxX = lm.x;
            if (lm.y < minY) minY = lm.y;
            if (lm.y > maxY) maxY = lm.y;
          });

          const faceW = maxX - minX;
          const faceH = maxY - minY;
          
          const scaleX = width / (faceW + 0.15);
          const scaleY = height / (faceH + 0.15);
          const scale = Math.min(scaleX, scaleY) * zoom;

          const centerX = minX + faceW / 2;
          const centerY = minY + faceH / 2;

          const screenCenterX = width / 2;
          const screenCenterY = height / 2;

          const cosY = Math.cos(yaw);
          const sinY = Math.sin(yaw);
          const cosP = Math.cos(pitch);
          const sinP = Math.sin(pitch);

          lms.forEach((lm: Landmark, idx: number) => {
            const xc = lm.x - centerX;
            const yc = lm.y - centerY;
            const zc = lm.z || 0;

            const xRotY = xc * cosY - zc * sinY;
            const zRotY = xc * sinY + zc * cosY;

            const yRotX = yc * cosP - zRotY * sinP;
            const zRotX = yc * sinP + zRotY * cosP;

            const screenX = xRotY * scale + screenCenterX;
            const screenY = yRotX * scale + screenCenterY;

            const imp = dl?.gnn_prediction?.node_importance?.[idx] ?? 0;

            projected.push({
              x: screenX,
              y: screenY,
              z: zRotX,
              importance: imp,
              idx
            });
          });
        } else {
          // Direct camera alignment mapping: matches the background video frame perfectly
          lms.forEach((lm: Landmark, idx: number) => {
            const screenX = dx + lm.x * drawW;
            const screenY = dy + lm.y * drawH;
            const zRotX = lm.z || 0;
            const imp = dl?.gnn_prediction?.node_importance?.[idx] ?? 0;

            projected.push({
              x: screenX,
              y: screenY,
              z: zRotX,
              importance: imp,
              idx
            });
          });
        }

        projectedPointsRef.current = projected.map(p => ({ x: p.x, y: p.y, idx: p.idx }));

        // ── 2. Draw Attention Heatmap (if enabled) ───────────────────────────
        if (showHeatmap && showXAI) {
          projected.forEach(p => {
            if (p.importance > 0.18) {
              const rad = p.importance * 20; // scaled down slightly
              const grad = ctx.createRadialGradient(p.x, p.y, 1, p.x, p.y, rad);
              grad.addColorStop(0, `rgba(239, 68, 68, ${p.importance * 0.45})`);
              grad.addColorStop(0.5, `rgba(245, 158, 11, ${p.importance * 0.2})`);
              grad.addColorStop(1, 'rgba(245, 158, 11, 0)');

              ctx.fillStyle = grad;
              ctx.beginPath();
              ctx.arc(p.x, p.y, rad, 0, 2 * Math.PI);
              ctx.fill();
            }
          });
        }

        // ── 3. Draw Connections / Edges ─────────────────────────────────────
        const connectionsToUse = showSkeleton 
          ? MESH_CONNECTIONS.filter(([p1]) => 
              LANDMARK_REGIONS.leftEye.includes(p1) || 
              LANDMARK_REGIONS.rightEye.includes(p1) || 
              LANDMARK_REGIONS.lips.includes(p1)
            )
          : MESH_CONNECTIONS;

        if (showMesh) {
          connectionsToUse.forEach(([p1, p2]) => {
            const pt1 = projected[p1];
            const pt2 = projected[p2];

            if (pt1 && pt2) {
              const avgImp = (pt1.importance + pt2.importance) / 2;

              // Thinner connection widths for cleaner aesthetic
              if (showGraph) {
                if (avgImp > 0.5) {
                  ctx.strokeStyle = `rgba(239, 68, 68, ${0.4 + avgImp * 0.6})`; 
                  ctx.lineWidth = 0.95;
                } else if (avgImp > 0.3) {
                  ctx.strokeStyle = `rgba(245, 158, 11, ${0.35 + avgImp * 0.5})`; 
                  ctx.lineWidth = 0.65;
                } else if (avgImp > 0.15) {
                  ctx.strokeStyle = `rgba(16, 185, 129, ${0.25 + avgImp * 0.4})`; 
                  ctx.lineWidth = 0.45;
                } else {
                  ctx.strokeStyle = 'rgba(99, 102, 241, 0.14)'; 
                  ctx.lineWidth = 0.3;
                }
              } else {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
                ctx.lineWidth = 0.3;
              }

              ctx.beginPath();
              ctx.moveTo(pt1.x, pt1.y);
              ctx.lineTo(pt2.x, pt2.y);
              ctx.stroke();

              // Moving information flow particle lines
              if (showGraph && avgImp > 0.3) {
                const particlePos = particleOffsetRef.current;
                const px = pt1.x + (pt2.x - pt1.x) * particlePos;
                const py = pt1.y + (pt2.y - pt1.y) * particlePos;

                ctx.fillStyle = avgImp > 0.5 ? '#ef4444' : '#f59f0b';
                ctx.beginPath();
                ctx.arc(px, py, 0.8, 0, 2 * Math.PI);
                ctx.fill();
              }
            }
          });
        }

        // ── 4. Draw Nodes / Landmarks ────────────────────────────────────────
        projected.forEach(p => {
          const isHighAct = p.importance > 0.5;
          const isActive = p.importance > 0.15;

          if (showXAI && isActive) {
            // Elegant, smaller node sizes
            const rad = isHighAct ? 1.8 + p.importance * 1.8 : 1.0 + p.importance * 1.2;
            
            ctx.fillStyle = isHighAct 
              ? `rgba(239, 68, 68, ${0.7 + p.importance * 0.3})`  
              : `rgba(245, 158, 11, ${0.6 + p.importance * 0.4})`; 

            ctx.beginPath();
            ctx.arc(p.x, p.y, rad, 0, 2 * Math.PI);
            ctx.fill();

            // Tight, micro pulsing ring (very subtle radar effect)
            ctx.strokeStyle = isHighAct ? 'rgba(239, 68, 68, 0.35)' : 'rgba(245, 158, 11, 0.25)';
            ctx.lineWidth = 0.45;
            ctx.beginPath();
            ctx.arc(p.x, p.y, rad + (Math.sin(Date.now() / 200) * 0.5 + 0.8), 0, 2 * Math.PI);
            ctx.stroke();
          } else {
            // Very small, crisp inactive node
            ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
            ctx.beginPath();
            ctx.arc(p.x, p.y, 0.55, 0, 2 * Math.PI);
            ctx.fill();
          }
        });

        // ── 5. Draw Eye Gaze Vectors ─────────────────────────────────────────
        if (latestResult?.eyes) {
          const lIris = projected[468]; // left iris center
          const rIris = projected[473]; // right iris center
          if (lIris && rIris) {
            let dx = 0, dy = 0;
            const gaze = latestResult.eyes.gaze_direction;
            if (gaze === 'left') dx = -20;
            else if (gaze === 'right') dx = 20;
            else if (gaze === 'up') dy = -20;
            else if (gaze === 'down') dy = 20;

            ctx.strokeStyle = 'rgba(34, 211, 238, 0.85)'; // Gaze vector color (cyan)
            ctx.lineWidth = 1.5;

            // Draw arrow from iris centers
            [lIris, rIris].forEach(iris => {
              ctx.beginPath();
              ctx.moveTo(iris.x, iris.y);
              ctx.lineTo(iris.x + dx, iris.y + dy);
              ctx.stroke();

              // Draw Arrowhead
              ctx.fillStyle = 'rgba(34, 211, 238, 0.85)';
              ctx.beginPath();
              ctx.arc(iris.x + dx, iris.y + dy, 2, 0, 2 * Math.PI);
              ctx.fill();
            });
          }
        }

        // ── 6. Draw 3D Head Pose Axes ────────────────────────────────────────
        if (latestResult?.head_pose) {
          const noseTip = projected[4]; // nose tip index
          if (noseTip) {
            const yawRad = latestResult.head_pose.yaw * (Math.PI / 180);
            const pitchRad = latestResult.head_pose.pitch * (Math.PI / 180);
            const rollRad = latestResult.head_pose.roll * (Math.PI / 180);

            // X-axis (Pitch - Red)
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            ctx.moveTo(noseTip.x, noseTip.y);
            ctx.lineTo(noseTip.x + Math.cos(rollRad) * 35, noseTip.y + Math.sin(rollRad) * 35);
            ctx.stroke();

            // Y-axis (Yaw - Green)
            ctx.strokeStyle = '#10b981';
            ctx.beginPath();
            ctx.moveTo(noseTip.x, noseTip.y);
            ctx.lineTo(noseTip.x - Math.sin(yawRad) * 35, noseTip.y - Math.cos(yawRad) * 35);
            ctx.stroke();

            // Z-axis (Roll - Blue/Cyan)
            ctx.strokeStyle = '#22d3ee';
            ctx.beginPath();
            ctx.moveTo(noseTip.x, noseTip.y);
            ctx.lineTo(noseTip.x + Math.sin(yawRad) * 20, noseTip.y + Math.cos(pitchRad) * 35);
            ctx.stroke();
          }
        }

        // ── 7. Render Debug Landmark IDs in Research Mode ────────────────────
        if (isResearchMode) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
          ctx.font = '5px monospace';
          projected.forEach((p, idx) => {
            if (idx % 4 === 0) { // space them out
              ctx.fillText(idx.toString(), p.x + 2, p.y + 1);
            }
          });
        }
      }

      // ── 8. Draw Hovered Node Tooltip Overlay ─────────────────────────────
      if (hoveredNode && isRunning) {
        ctx.strokeStyle = '#fb7185'; // rose target reticle
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.arc(hoveredNode.screenX, hoveredNode.screenY, 7, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(hoveredNode.screenX - 12, hoveredNode.screenY); ctx.lineTo(hoveredNode.screenX + 12, hoveredNode.screenY);
        ctx.moveTo(hoveredNode.screenX, hoveredNode.screenY - 12); ctx.lineTo(hoveredNode.screenX, hoveredNode.screenY + 12);
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isRunning, latestResult, showCamera, showMesh, showGraph, showHeatmap, showSkeleton, showXAI, yaw, pitch, zoom, hoveredNode, isResearchMode]);

  // Handle Drag / Rotation Mouse events
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isDragging) {
      const deltaX = e.clientX - dragStartRef.current.x;
      const deltaY = e.clientY - dragStartRef.current.y;
      setYaw(prev => prev + deltaX * 0.008);
      setPitch(prev => prev + deltaY * 0.008);
      dragStartRef.current = { x: e.clientX, y: e.clientY };
    } else if (latestResult?.deep_learning?.landmarks) {
      // Find nearest projected landmark to mouse cursor
      const lms = latestResult.deep_learning.landmarks;
      const ptList = projectedPointsRef.current;
      
      let closestIdx = -1;
      let minDistance = 14; // max distance limit

      ptList.forEach(pt => {
        const dx = pt.x - x;
        const dy = pt.y - y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDistance) {
          minDistance = dist;
          closestIdx = pt.idx;
        }
      });

      if (closestIdx !== -1) {
        const rawLm = lms[closestIdx];
        const screenPt = ptList.find(p => p.idx === closestIdx);
        const imp = latestResult.deep_learning.gnn_prediction?.node_importance?.[closestIdx] ?? 0;

        setHoveredNode({
          id: closestIdx,
          region: getRegionName(closestIdx),
          importance: imp,
          x: rawLm.x,
          y: rawLm.y,
          z: rawLm.z || 0,
          screenX: screenPt?.x ?? 0,
          screenY: screenPt?.y ?? 0,
        });
      } else {
        setHoveredNode(null);
      }
    }
  };

  const handleMouseUpOrLeave = () => {
    setIsDragging(false);
  };

  // Zoom control
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    setZoom(prev => Math.max(0.4, Math.min(3.0, prev - e.deltaY * 0.001)));
  };

  const reset3D = () => {
    setYaw(0);
    setPitch(0);
    setZoom(1.0);
    setHoveredNode(null);
  };

  return (
    <div className="bg-dark-900 border border-dark-600 rounded-2xl p-4 flex flex-col gap-4 shadow-2xl relative overflow-hidden">
      {/* HUD Header */}
      <div className="flex items-center justify-between z-10">
        <div className="flex items-center gap-2">
          <Rotate3d size={18} className="text-violet-400 animate-pulse" />
          <span className="text-xs font-bold font-mono uppercase tracking-wider text-slate-300">
            3D Graph Projection HUD
          </span>
        </div>
        <div className="flex items-center gap-3">
          {/* Mode Switchers */}
          <div className="flex items-center bg-slate-950/60 border border-dark-600/50 rounded-lg p-0.5 text-[10px] font-mono">
            <button 
              onClick={() => setResearchMode(false)}
              className={`px-2.5 py-1 rounded transition-colors ${!isResearchMode ? 'bg-violet-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Standard
            </button>
            <button 
              onClick={() => setResearchMode(true)}
              className={`px-2.5 py-1 rounded transition-colors ${isResearchMode ? 'bg-violet-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Research
            </button>
          </div>
          
          <button 
            onClick={reset3D}
            className="p-1.5 bg-dark-800/80 border border-dark-600/40 rounded-lg text-dark-300 hover:text-white transition-colors"
            title="Reset Orbit Camera"
          >
            <Rotate3d size={14} />
          </button>
        </div>
      </div>

      {/* Primary Canvas Container */}
      <div className="relative rounded-xl overflow-hidden bg-slate-950/80 border border-dark-600/60 aspect-video group cursor-grab active:cursor-grabbing">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUpOrLeave}
          onMouseLeave={handleMouseUpOrLeave}
          onWheel={handleWheel}
          className="absolute inset-0 w-full h-full object-cover"
        />

        {/* Video reference element used by canvas context drawImage (kept hidden) */}
        <video
          ref={videoRef}
          className="hidden"
          muted
          playsInline
          autoPlay
        />

        {/* Alert banners inside HUD */}
        {activeAlerts.length > 0 && (
          <div className="absolute top-3 left-3 right-3 space-y-1.5 pointer-events-none">
            {activeAlerts.slice(0, 2).map((alert, i) => (
              <div key={i} className="bg-risk-critical/25 border border-risk-critical/40 backdrop-blur-md rounded-lg px-3 py-2 animate-slide-in-right">
                <span className="text-risk-critical text-[11px] font-mono font-semibold flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-risk-critical animate-ping" />
                  CRITICAL: {alert}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Node Hover Tooltip Card */}
        {hoveredNode && isRunning && (
          <div 
            className="absolute z-30 bg-slate-950/90 border border-rose-500/40 backdrop-blur-md rounded-xl p-3 shadow-2xl font-mono text-[10px] text-slate-300 pointer-events-none select-none"
            style={{ 
              left: `${Math.min(hoveredNode.screenX + 15, canvasRef.current ? canvasRef.current.clientWidth - 150 : 200)}px`,
              top: `${Math.min(hoveredNode.screenY + 15, canvasRef.current ? canvasRef.current.clientHeight - 130 : 200)}px`
            }}
          >
            <div className="text-[11px] font-extrabold text-rose-400 border-b border-rose-500/20 pb-1 mb-1.5 flex items-center justify-between">
              <span>NODE #{hoveredNode.id}</span>
              <span className="text-[9px] bg-rose-500/10 px-1 rounded">GNN Node</span>
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between"><span>Region:</span><span className="text-slate-100 font-bold">{hoveredNode.region}</span></div>
              <div className="flex justify-between"><span>Attribution:</span><span className="text-violet-400 font-bold">{Math.round(hoveredNode.importance * 100)}%</span></div>
              <div className="flex justify-between"><span>X coordinate:</span><span className="text-slate-400">{hoveredNode.x.toFixed(4)}</span></div>
              <div className="flex justify-between"><span>Y coordinate:</span><span className="text-slate-400">{hoveredNode.y.toFixed(4)}</span></div>
              <div className="flex justify-between"><span>Z coordinate:</span><span className="text-slate-400">{hoveredNode.z.toFixed(4)}</span></div>
            </div>
          </div>
        )}

        {/* HUD Statistics Overlay */}
        <div className="absolute bottom-3 left-3 bg-slate-950/70 border border-dark-600/30 backdrop-blur-md p-2.5 rounded-lg pointer-events-none font-mono text-[9px] text-slate-400 space-y-1">
          <div className="flex justify-between gap-4"><span>FRAME YAW:</span><span className="text-emerald-400 font-bold">{(yaw * (180 / Math.PI)).toFixed(1)}°</span></div>
          <div className="flex justify-between gap-4"><span>FRAME PITCH:</span><span className="text-emerald-400 font-bold">{(pitch * (180 / Math.PI)).toFixed(1)}°</span></div>
          <div className="flex justify-between gap-4"><span>ZOOM SCALE:</span><span className="text-violet-400 font-bold">{Math.round(zoom * 100)}%</span></div>
        </div>

        {/* Orbit Instructions Overlay */}
        <div className="absolute top-3 right-3 bg-slate-950/70 border border-dark-600/30 backdrop-blur-md px-2 py-1 rounded text-[9px] text-slate-400 font-mono pointer-events-none">
          Drag to Rotate · Scroll to Zoom
        </div>

        {/* Scan corner brackets */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-violet-500/50 rounded-tl-lg" />
          <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-violet-500/50 rounded-tr-lg" />
          <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-violet-500/50 rounded-bl-lg" />
          <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-violet-500/50 rounded-br-lg" />
        </div>
      </div>

      {/* Control Panel Strip (Microsoft/Azure AI minimal grid design) */}
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 bg-slate-950/30 border border-dark-600/20 rounded-xl p-2 z-10">
        <button
          onClick={() => setShowCamera(!showCamera)}
          className={`py-1.5 rounded-lg border text-[10px] font-bold font-mono transition-all uppercase ${
            showCamera ? 'bg-violet-500/10 border-violet-500/40 text-violet-400' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          Camera
        </button>
        <button
          onClick={() => setShowMesh(!showMesh)}
          className={`py-1.5 rounded-lg border text-[10px] font-bold font-mono transition-all uppercase ${
            showMesh ? 'bg-violet-500/10 border-violet-500/40 text-violet-400' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          Mesh
        </button>
        <button
          onClick={() => setShowGraph(!showGraph)}
          className={`py-1.5 rounded-lg border text-[10px] font-bold font-mono transition-all uppercase ${
            showGraph ? 'bg-violet-500/10 border-violet-500/40 text-violet-400' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          GNN Graph
        </button>
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className={`py-1.5 rounded-lg border text-[10px] font-bold font-mono transition-all uppercase ${
            showHeatmap ? 'bg-violet-500/10 border-violet-500/40 text-violet-400' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          Heatmap
        </button>
        <button
          onClick={() => setShowSkeleton(!showSkeleton)}
          className={`py-1.5 rounded-lg border text-[10px] font-bold font-mono transition-all uppercase ${
            showSkeleton ? 'bg-violet-500/10 border-violet-500/40 text-violet-400' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          Skeleton
        </button>
        <button
          onClick={() => setShowXAI(!showXAI)}
          className={`py-1.5 rounded-lg border text-[10px] font-bold font-mono transition-all uppercase ${
            showXAI ? 'bg-violet-500/10 border-violet-500/40 text-violet-400' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          XAI Layer
        </button>
      </div>

      {/* Start/Stop analysis control */}
      <div className="flex gap-2">
        {!isRunning ? (
          <button
            onClick={start}
            className="flex-1 bg-violet-600 hover:bg-violet-500 text-white font-bold py-2.5 px-4 rounded-lg
                       transition-all duration-200 flex items-center justify-center gap-2 text-xs font-mono uppercase tracking-wider
                       shadow-[0_0_20px_rgba(124,58,237,0.3)] hover:shadow-[0_0_30px_rgba(124,58,237,0.5)]"
          >
            <Camera size={14} />
            Initialize Live Capture
          </button>
        ) : (
          <button
            onClick={stop}
            className="flex-1 bg-risk-critical/20 hover:bg-risk-critical/30 border border-risk-critical/40 
                       text-risk-critical font-bold py-2.5 px-4 rounded-lg transition-all duration-200
                       flex items-center justify-center gap-2 text-xs font-mono uppercase tracking-wider"
          >
            <CameraOff size={14} />
            Deactivate Camera Feed
          </button>
        )}
      </div>
    </div>
  );
};
