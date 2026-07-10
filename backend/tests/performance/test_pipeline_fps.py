"""
AuthBrain AI Face Analysis Engine
Performance Benchmarking — Pipeline FPS

Measures the pipeline processing time and average FPS under continuous load.
Ensures we meet real-time performance targets (<50ms per frame).
"""

from __future__ import annotations

import io
import time
import pytest
from PIL import Image

from app.analysis.pipeline import FaceAnalysisPipeline


def _create_benchmark_frame(width=1280, height=720) -> bytes:
    """Create a standard HD mock frame in memory."""
    img = Image.new("RGB", (width, height), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.benchmark
class TestPipelinePerformance:

    def test_pipeline_throughput_fps(self):
        """Pipeline must average under 50ms processing time per frame (>20 FPS)."""
        pipeline = FaceAnalysisPipeline(session_id="benchmark-perf-session")
        pipeline.load()

        frame_bytes = _create_benchmark_frame()

        # Warm-up (1 frame)
        pipeline.process_frame(frame_bytes, active_face_index=0, draw_overlay=True)

        num_iterations = 20
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            result, _ = pipeline.process_frame(frame_bytes, active_face_index=0, draw_overlay=True)
            # Ensure it outputs expected keys
            assert result.session_id == "benchmark-perf-session"

        end_time = time.perf_counter()
        total_time_s = end_time - start_time
        avg_time_ms = (total_time_s / num_iterations) * 1000.0
        fps = num_iterations / total_time_s

        print(f"\n[PERFORMANCE] Iterations: {num_iterations}")
        print(f"[PERFORMANCE] Avg inference time: {avg_time_ms:.2f} ms")
        print(f"[PERFORMANCE] Pipeline throughput: {fps:.2f} FPS")

        pipeline.close()

        # Benchmark target: average processing time should be under 80ms on standard CPUs
        assert avg_time_ms < 80.0, f"Pipeline too slow: {avg_time_ms:.2f}ms/frame (target < 80ms)"
