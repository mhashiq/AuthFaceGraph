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
  lips: [
    // Lips Upper Outer
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    // Lips Lower Outer
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    // Lips Upper Inner
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    // Lips Lower Inner
    95, 88, 178, 87, 14, 317, 402, 318, 324
  ],
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
  // Lips / Mouth Outer
  [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270],
  [270, 409], [409, 291], [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314],
  [314, 405], [405, 321], [321, 375], [375, 291],
  // Lips / Mouth Inner
  [78, 191], [191, 80], [80, 81], [81, 82], [82, 13], [13, 312], [312, 311], [311, 310], [310, 415], [415, 308],
  [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324], [324, 308]
];

const SKELETON_CONNECTIONS = [
  [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19],
  [168, 33], [168, 362], [33, 133], [362, 263],
  [61, 291], [291, 308], [308, 324], [324, 78], [78, 61],
  [0, 17], [17, 84], [84, 181], [181, 314], [314, 405], [405, 321],
  [10, 152], [152, 234], [234, 454], [454, 323]
];

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const getNodeRadius = (importance: number, kind: 'mesh' | 'graph' | 'heatmap' | 'xai') => {
  if (kind === 'graph') return 2;
  if (kind === 'heatmap') return 2;
  if (kind === 'xai') return importance > 0.65 ? 3 : importance > 0.35 ? 2.5 : 2;
  return 1.75;
};

const getImportanceColor = (importance: number) => {
  if (importance > 0.8) return 'rgba(239, 68, 68, 1)';
  if (importance > 0.65) return 'rgba(249, 115, 22, 1)';
  if (importance > 0.45) return 'rgba(234, 179, 8, 1)';
  if (importance > 0.25) return 'rgba(34, 197, 94, 1)';
  return 'rgba(59, 130, 246, 1)';
};

const getRegionCenter = (landmarks: Landmark[], indices: number[]) => {
  const pts = indices.map(i => landmarks[i]).filter(Boolean);
  if (pts.length === 0) return null;
  const sum = pts.reduce((acc, lm) => ({ x: acc.x + lm.x, y: acc.y + lm.y }), { x: 0, y: 0 });
  return { x: sum.x / pts.length, y: sum.y / pts.length };
};

const TOGGLE_BASE = 'inline-flex items-center justify-center gap-1 h-7 px-2.5 rounded-full border text-[10px] font-mono font-semibold transition-all duration-150';

export const PrimaryAIVisualizer: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { videoRef, isRunning, cameraError, start, stop } = useFaceAnalysis();
  const latestResult = useAnalysisStore(s => s.latestResult);
  const activeAlerts = useAnalysisStore(s => s.activeAlerts);
  const [isResearchMode, setResearchMode] = useState(false);
  const [showDebug, setShowDebug] = useState(false);

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

  // Temporal smoothing for GNN node importance values (EMA)
  const smoothedImportanceRef = useRef<number[]>(new Array(478).fill(0));

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

      let drawW = width;
      let drawH = height;
      let dx = 0;
      let dy = 0;

      if (video && video.videoWidth > 0 && video.videoHeight > 0) {
        const vWidth = video.videoWidth;
        const vHeight = video.videoHeight;
        const videoRatio = vWidth / vHeight;
        const canvasRatio = width / height;

        if (canvasRatio > videoRatio) {
          drawW = width;
          drawH = width / videoRatio;
          dx = 0;
          dy = (height - drawH) / 2;
        } else {
          drawW = height * videoRatio;
          drawH = height;
          dx = (width - drawW) / 2;
          dy = 0;
        }
      }

      if (showCamera && isRunning && video && video.readyState >= 2) {
        ctx.save();
        ctx.globalAlpha = 0.45;
        ctx.filter = 'grayscale(60%) contrast(120%)';
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
      const lms = latestResult?.landmarks || dl?.landmarks || latestResult?.deep_learning?.landmarks;

      if (isRunning && lms && lms.length > 0) {
        // Use the backend bounding box for size/debug only. Landmark coordinates themselves
        // are normalized frame coordinates and must be plotted directly in canvas space.
        const faceBox = latestResult?.bounding_box;
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        lms.forEach((lm: Landmark) => {
          if (lm.x < minX) minX = lm.x;
          if (lm.y < minY) minY = lm.y;
          if (lm.x > maxX) maxX = lm.x;
          if (lm.y > maxY) maxY = lm.y;
        });

        const faceW = Math.max(0.001, (faceBox?.width ?? (maxX - minX)));
        const faceH = Math.max(0.001, (faceBox?.height ?? (maxY - minY)));
        const faceCenterX = faceBox ? (faceBox.x + faceBox.width / 2) : (minX + (maxX - minX) / 2);
        const faceCenterY = faceBox ? (faceBox.y + faceBox.height / 2) : (minY + (maxY - minY) / 2);

        // Compute smooth node importance weights using exponential moving average (EMA)
        const alpha = 0.25;
        const smoothedImps = new Array(lms.length).fill(0);
        lms.forEach((lm: Landmark, idx: number) => {
          const rawImp = dl?.gnn_prediction?.node_importance?.[idx] ?? 0;
          if (smoothedImportanceRef.current[idx] === undefined) {
            smoothedImportanceRef.current[idx] = 0;
          }
          smoothedImportanceRef.current[idx] = alpha * rawImp + (1 - alpha) * (smoothedImportanceRef.current[idx] || 0);
          smoothedImps[idx] = smoothedImportanceRef.current[idx];
        });

        const projected: { x: number; y: number; z: number; importance: number; idx: number }[] = [];
        const faceFootprintPx = showCamera
          ? Math.max(faceW * drawW, faceH * drawH)
          : Math.max(faceW * width, faceH * height);
        const overlayScale = Math.max(1.3, Math.min(4.8, faceFootprintPx / 70));

        const faceRect = {
          x: showCamera ? dx + (faceBox?.x ?? minX) * drawW : (faceBox?.x ?? minX) * width,
          y: showCamera ? dy + (faceBox?.y ?? minY) * drawH : (faceBox?.y ?? minY) * height,
          w: showCamera ? faceW * drawW : faceW * width,
          h: showCamera ? faceH * drawH : faceH * height,
        };

        const toScreen = (lm: Landmark) => ({
          x: showCamera ? dx + lm.x * drawW : lm.x * width,
          y: showCamera ? dy + lm.y * drawH : lm.y * height,
        });

        const meshConnections = MESH_CONNECTIONS;
        const skeletonConnections = SKELETON_CONNECTIONS;
        const hasEdgeAttention = Boolean(dl?.gnn_prediction?.edge_index && dl.gnn_prediction.edge_index.length === 2);
        const edgeAttentionMap = new Map<string, number>();
        let maxAtt = 0.001;
        if (hasEdgeAttention) {
          const edgeIndex = dl!.gnn_prediction!.edge_index!;
          const edgeAttention = dl!.gnn_prediction!.edge_attention || [];
          const numEdges = edgeIndex[0].length;
          for (let i = 0; i < numEdges; i++) {
            const u = edgeIndex[0][i];
            const v = edgeIndex[1][i];
            const attVal = edgeAttention[i] ?? 0.05;
            if (attVal > maxAtt) maxAtt = attVal;
            const key = u < v ? `${u}-${v}` : `${v}-${u}`;
            edgeAttentionMap.set(key, Math.max(edgeAttentionMap.get(key) || 0, attVal));
          }
        }

        ctx.save();
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.28)';
        ctx.lineWidth = 1.25 * overlayScale;
        ctx.setLineDash([6 * overlayScale, 5 * overlayScale]);
        ctx.strokeRect(faceRect.x, faceRect.y, faceRect.w, faceRect.h);
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(168, 85, 247, 0.08)';
        ctx.fillRect(faceRect.x, faceRect.y, faceRect.w, faceRect.h);
        ctx.restore();

        lms.forEach((lm: Landmark, idx: number) => {
          const screen = toScreen(lm);
          projected.push({ x: screen.x, y: screen.y, z: lm.z || 0, importance: smoothedImps[idx], idx });
        });

        projectedPointsRef.current = projected.map(p => ({ x: p.x, y: p.y, idx: p.idx }));

        // ── 2. Camera / Bounding Box layers ──────────────────────────────────
        // Camera is already drawn above. Bounding box remains visible as a separate layer.

        // ── 3. Face Mesh layer ───────────────────────────────────────────────
        if (showMesh) {
          meshConnections.forEach(([p1, p2]) => {
            const pt1 = projected[p1];
            const pt2 = projected[p2];
            if (!pt1 || !pt2) return;

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            ctx.moveTo(pt1.x, pt1.y);
            ctx.lineTo(pt2.x, pt2.y);
            ctx.stroke();
          });

          projected.forEach(p => {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
            ctx.beginPath();
            ctx.arc(p.x, p.y, 1.75, 0, 2 * Math.PI);
            ctx.fill();
          });
        }

        // ── 4. Skeleton layer ────────────────────────────────────────────────
        if (showSkeleton) {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
          ctx.lineWidth = 0.8;
          skeletonConnections.forEach(([p1, p2]) => {
            const pt1 = projected[p1];
            const pt2 = projected[p2];
            if (!pt1 || !pt2) return;
            ctx.beginPath();
            ctx.moveTo(pt1.x, pt1.y);
            ctx.lineTo(pt2.x, pt2.y);
            ctx.stroke();
          });
        }

        // ── 5. GNN Graph layer ───────────────────────────────────────────────
        if (showGraph) {
          const edgeAlpha = 0.5;
          const graphNodes = projected;
          const edgeIndex = dl?.gnn_prediction?.edge_index;
          const edgeAttention = dl?.gnn_prediction?.edge_attention || [];
          if (edgeIndex && edgeIndex.length === 2) {
            const numEdges = edgeIndex[0].length;
            for (let i = 0; i < numEdges; i++) {
              const u = edgeIndex[0][i];
              const v = edgeIndex[1][i];
              const pt1 = graphNodes[u];
              const pt2 = graphNodes[v];
              if (!pt1 || !pt2) continue;

              const att = edgeAttention[i] ?? 0.05;
              ctx.strokeStyle = `rgba(59, 130, 246, ${clamp(edgeAlpha + att * 0.4, 0.5, 0.95)})`;
              ctx.lineWidth = att > 0.35 ? 1.5 : 1.0;
              ctx.beginPath();
              ctx.moveTo(pt1.x, pt1.y);
              ctx.lineTo(pt2.x, pt2.y);
              ctx.stroke();
            }
          }

          graphNodes.forEach(p => {
            ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
            ctx.beginPath();
            ctx.arc(p.x, p.y, 2, 0, 2 * Math.PI);
            ctx.fill();
          });
        }

        // ── 6. Heatmap layer ────────────────────────────────────────────────
        if (showHeatmap) {
          projected.forEach(p => {
            if (p.importance <= 0.02) return;
            const opacity = clamp(0.08 + p.importance * 0.55, 0.08, 0.65);
            ctx.fillStyle = getImportanceColor(p.importance).replace('1)', `${opacity})`);
            ctx.beginPath();
            ctx.arc(p.x, p.y, 2, 0, 2 * Math.PI);
            ctx.fill();
          });
        }

        // ── 7. XAI layer ────────────────────────────────────────────────────
        if (showXAI) {
          const ranked = projected
            .map(p => ({ ...p, score: Math.max(p.importance, dl?.gnn_prediction?.node_importance?.[p.idx] ?? 0) }))
            .sort((a, b) => b.score - a.score)
            .slice(0, 16);

          const importantIndices = new Set(ranked.map(p => p.idx));
          const importantRegions = [
            { name: 'nose', indices: LANDMARK_REGIONS.nose },
            { name: 'leftEye', indices: LANDMARK_REGIONS.leftEye },
            { name: 'rightEye', indices: LANDMARK_REGIONS.rightEye },
            { name: 'lips', indices: LANDMARK_REGIONS.lips },
            { name: 'leftEyebrow', indices: LANDMARK_REGIONS.leftEyebrow },
            { name: 'rightEyebrow', indices: LANDMARK_REGIONS.rightEyebrow },
          ]
            .map(region => ({ ...region, center: getRegionCenter(lms, region.indices) }))
            .filter(region => region.center !== null);

          const xaiEdges = dl?.gnn_prediction?.edge_index;
          if (xaiEdges && xaiEdges.length === 2) {
            const edgeAttention = dl?.gnn_prediction?.edge_attention || [];
            for (let i = 0; i < xaiEdges[0].length; i++) {
              const u = xaiEdges[0][i];
              const v = xaiEdges[1][i];
              if (!importantIndices.has(u) && !importantIndices.has(v)) continue;
              const pt1 = projected[u];
              const pt2 = projected[v];
              if (!pt1 || !pt2) continue;

              const att = edgeAttention[i] ?? 0.1;
              ctx.strokeStyle = `rgba(245, 158, 11, ${clamp(0.25 + att * 0.75, 0.25, 0.9)})`;
              ctx.lineWidth = 1.5;
              ctx.beginPath();
              ctx.moveTo(pt1.x, pt1.y);
              ctx.lineTo(pt2.x, pt2.y);
              ctx.stroke();
            }
          }

          ranked.forEach(p => {
            ctx.fillStyle = 'rgba(245, 158, 11, 0.85)';
            ctx.beginPath();
            ctx.arc(p.x, p.y, getNodeRadius(p.score, 'xai'), 0, 2 * Math.PI);
            ctx.fill();

            ctx.strokeStyle = 'rgba(245, 158, 11, 0.25)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(p.x, p.y, getNodeRadius(p.score, 'xai') + (Math.sin(Date.now() / 180) * 0.4 + 0.8), 0, 2 * Math.PI);
            ctx.stroke();
          });

          importantRegions.forEach(region => {
            if (!region.center) return;
            const screen = toScreen({ x: region.center.x, y: region.center.y, z: 0 } as Landmark);
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.15)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(screen.x, screen.y, 22, 0, 2 * Math.PI);
            ctx.stroke();
          });
        }

        // ── 8. Debug overlay ───────────────────────────────────────────────
        if (showDebug) {
          const debugX = 16;
          const debugY = 16;
          const debugLines = [
            `FPS: ${latestResult?.fps?.toFixed(1) ?? '—'}`,
            `Inference: ${latestResult?.inference_time_ms?.toFixed(1) ?? '—'} ms`,
            `Frame: ${latestResult?.frame_width ?? 0}x${latestResult?.frame_height ?? 0}`,
            `ROI: ${Math.round(faceRect.x)}, ${Math.round(faceRect.y)}, ${Math.round(faceRect.w)}, ${Math.round(faceRect.h)}`,
            `BBox: ${faceBox ? `${faceBox.x.toFixed(3)}, ${faceBox.y.toFixed(3)}, ${faceBox.width.toFixed(3)}, ${faceBox.height.toFixed(3)}` : 'n/a'}`,
            `Nodes: ${projected.length} | Edges: ${meshConnections.length}`,
            `Nose tip idx 1: ${projected[1] ? `${projected[1].x.toFixed(1)}, ${projected[1].y.toFixed(1)}` : 'n/a'}`,
          ];

          ctx.save();
          ctx.fillStyle = 'rgba(2, 6, 23, 0.8)';
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.roundRect(debugX - 8, debugY - 8, 470, 160, 10);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = 'rgba(226, 232, 240, 0.95)';
          ctx.font = '11px monospace';
          debugLines.forEach((line, index) => ctx.fillText(line, debugX, debugY + 14 + index * 18));

          projected.forEach((p, idx) => {
            if (idx % 12 === 0) {
              const lm = lms[idx];
              ctx.fillStyle = 'rgba(226, 232, 240, 0.8)';
              ctx.fillText(`${idx}:${lm.x.toFixed(3)},${lm.y.toFixed(3)}`, p.x + 4, p.y - 4);
            }
          });
          ctx.restore();
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
  }, [isRunning, latestResult, showCamera, showMesh, showGraph, showHeatmap, showSkeleton, showXAI, showDebug, hoveredNode, isResearchMode]);

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
      <div className="flex flex-wrap gap-2 bg-slate-950/30 border border-dark-600/20 rounded-xl p-2 z-10">
        <button
          onClick={() => setShowCamera(!showCamera)}
          className={`${TOGGLE_BASE} ${
            showCamera ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          <Camera size={11} /> Camera <span className="opacity-80">{showCamera ? 'ON' : 'OFF'}</span>
        </button>
        <button
          onClick={() => setShowMesh(!showMesh)}
          className={`${TOGGLE_BASE} ${
            showMesh ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          <span>•</span> Mesh <span className="opacity-80">{showMesh ? 'ON' : 'OFF'}</span>
        </button>
        <button
          onClick={() => setShowGraph(!showGraph)}
          className={`${TOGGLE_BASE} ${
            showGraph ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          <Layout size={11} /> Graph <span className="opacity-80">{showGraph ? 'ON' : 'OFF'}</span>
        </button>
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className={`${TOGGLE_BASE} ${
            showHeatmap ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          <span>🔥</span> Heatmap <span className="opacity-80">{showHeatmap ? 'ON' : 'OFF'}</span>
        </button>
        <button
          onClick={() => setShowSkeleton(!showSkeleton)}
          className={`${TOGGLE_BASE} ${
            showSkeleton ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          <span>╱</span> Skeleton <span className="opacity-80">{showSkeleton ? 'ON' : 'OFF'}</span>
        </button>
        <button
          onClick={() => setShowXAI(!showXAI)}
          className={`${TOGGLE_BASE} ${
            showXAI ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          <span>🧠</span> XAI <span className="opacity-80">{showXAI ? 'ON' : 'OFF'}</span>
        </button>
        <button
          onClick={() => setShowDebug(!showDebug)}
          className={`${TOGGLE_BASE} ${
            showDebug ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300' : 'bg-transparent border-dark-600/40 text-dark-400'
          }`}
        >
          <span>🐞</span> Debug <span className="opacity-80">{showDebug ? 'ON' : 'OFF'}</span>
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
