import sys
import os
import cv2
import time
import numpy as np

# Ensure backend imports work
sys.path.append("/Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend")

import app
from app.analysis.pipeline import FaceAnalysisPipeline
from app.analysis.face_detector import FaceDetector
from app.dl.engine import DLEngine

# Paths to generated test images
TEST_IMAGES = {
    "happy": "/Users/mdmehedihassan/.gemini/antigravity-ide/brain/90dd8f33-233f-4ad4-bb24-75f2f1329f30/happy_face_1783690622508.png",
    "sad": "/Users/mdmehedihassan/.gemini/antigravity-ide/brain/90dd8f33-233f-4ad4-bb24-75f2f1329f30/sad_face_1783690636838.png",
    "anger": "/Users/mdmehedihassan/.gemini/antigravity-ide/brain/90dd8f33-233f-4ad4-bb24-75f2f1329f30/angry_face_1783690652952.png",
    "fear": "/Users/mdmehedihassan/.gemini/antigravity-ide/brain/90dd8f33-233f-4ad4-bb24-75f2f1329f30/fear_face_1783690669036.png",
    "surprise": "/Users/mdmehedihassan/.gemini/antigravity-ide/brain/90dd8f33-233f-4ad4-bb24-75f2f1329f30/surprise_face_1783690688522.png",
    "disgust": "/Users/mdmehedihassan/.gemini/antigravity-ide/brain/90dd8f33-233f-4ad4-bb24-75f2f1329f30/disgust_face_1783690707585.png",
    "neutral": "/Users/mdmehedihassan/.gemini/antigravity-ide/brain/90dd8f33-233f-4ad4-bb24-75f2f1329f30/neutral_face_1783690725963.png",
    "contempt": "/Users/mdmehedihassan/.gemini/antigravity-ide/brain/90dd8f33-233f-4ad4-bb24-75f2f1329f30/contempt_face_1783690747539.png",
}

def verify_pipeline():
    print("Initializing FaceDetector and DLEngine...")
    detector = FaceDetector()
    detector.load_model()
    
    engine = DLEngine(session_id="verification_session")
    engine.load()
    
    results = []
    
    for expected_emotion, path in TEST_IMAGES.items():
        print(f"\nProcessing {expected_emotion} face from {path}...")
        if not os.path.exists(path):
            print(f"ERROR: File {path} not found!")
            continue
            
        frame = cv2.imread(path)
        if frame is None:
            print(f"ERROR: Failed to load {path}!")
            continue
            
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Face detection
        faces = detector.detect(frame_rgb, w, h)
        if not faces:
            print(f"ERROR: Face not detected in {expected_emotion} image!")
            results.append({
                "expected": expected_emotion,
                "status": "Face not detected",
                "actual": "N/A",
                "confidence": 0.0,
                "latency_ms": 0.0,
            })
            continue
            
        face_data = faces[0]
        
        # 2. DLEngine processing
        t_start = time.perf_counter()
        res_dict = engine.process(
            face_data=face_data,
            frame_bgr=frame,
            frame_width=w,
            frame_height=h,
            timestamp_ms=0.0,
            quality_score=1.0,
        )
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        
        if not res_dict:
            print(f"ERROR: DL engine execution failed!")
            results.append({
                "expected": expected_emotion,
                "status": "DL processing failed",
                "actual": "N/A",
                "confidence": 0.0,
                "latency_ms": latency_ms,
            })
            continue
            
        ensemble = res_dict["emotion_ensemble"]
        actual_emotion = ensemble["final_emotion"]
        confidence = ensemble["confidence"]
        
        # Check active models list
        models_used = res_dict["models_used"]
        print(f"SUCCESS: Expected={expected_emotion}, Actual={actual_emotion}, Confidence={confidence * 100:.2f}%, Models={models_used}")
        
        results.append({
            "expected": expected_emotion,
            "status": "Success",
            "actual": actual_emotion,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "models": models_used,
            "probabilities": ensemble["probabilities"]
        })
        
    # Generate verification report
    report_content = "# Verification Validation Report\n\n"
    report_content += "This report evaluates the accuracy of the backend emotion recognition pipeline on generated high-quality front portraits.\n\n"
    report_content += "## System Information\n"
    report_content += "- **Primary Model**: HSEmotion ONNX (`enet_b2_8`)\n"
    report_content += "- **Ensemble Config**: `[\"hsemotion\"]` (EfficientFace deactivated due to private Hugging Face checkpoint weights)\n"
    report_content += "- **Target Device**: CPU (Mac Host)\n\n"
    report_content += "## Validation Results\n\n"
    report_content += "| Expected Label | Actual Prediction | Confidence | Status | Inference Time | Models Used |\n"
    report_content += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    success_count = 0
    total_count = 0
    
    for r in results:
        total_count += 1
        conf_str = f"{r['confidence'] * 100:.1f}%" if r['status'] == "Success" else "N/A"
        latency_str = f"{r['latency_ms']:.1f} ms"
        models_str = ", ".join(r.get('models', [])) if r['status'] == "Success" else "N/A"
        
        report_content += f"| **{r['expected'].capitalize()}** | {r['actual'].capitalize() if r['status'] == 'Success' else 'N/A'} | {conf_str} | {r['status']} | {latency_str} | {models_str} |\n"
        
        if r['status'] == "Success" and r['expected'] == r['actual']:
            success_count += 1
            
    accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
    report_content += f"\n**Overall Accuracy**: {success_count}/{total_count} ({accuracy:.1f}%)\n\n"
    
    report_content += "## Probabilities Grid\n\n"
    report_content += "| Expected Label | " + " | ".join([e.capitalize() for e in app.dl.emotion.hsemotion.EMOTION_LABELS]) + " |\n"
    report_content += "| :--- | " + " | ".join([":---" for _ in app.dl.emotion.hsemotion.EMOTION_LABELS]) + " |\n"
    
    for r in results:
        if r['status'] == "Success":
            row_str = f"| **{r['expected'].capitalize()}** | "
            row_str += " | ".join([f"{r['probabilities'].get(e, 0.0) * 100:.1f}%" for e in app.dl.emotion.hsemotion.EMOTION_LABELS])
            row_str += " |\n"
            report_content += row_str
            
    # Write report to artifact folder
    report_path = "/Users/mdmehedihassan/.gemini/antigravity-ide/brain/90dd8f33-233f-4ad4-bb24-75f2f1329f30/validation_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nValidation report successfully saved to {report_path}")

if __name__ == "__main__":
    verify_pipeline()
