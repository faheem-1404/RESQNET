import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import os
import torch

# Monkey-patch torch.load to bypass PyTorch 2.6's weights_only=True default
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

def draw_futuristic_ui(frame, fps, flood_percentage, severity, confidences):
    """Draw a futuristic HUD over the frame."""
    h, w = frame.shape[:2]
    
    # HUD colors
    hud_color = (255, 255, 0) if severity != "CRITICAL" else (0, 0, 255)
    bg_color = (15, 15, 15)
    
    # Top bar background (semi-transparent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), bg_color, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Draw corners (Futuristic brackets)
    t = 20
    thick = 3
    # Top left
    cv2.line(frame, (10, 10), (10+t, 10), hud_color, thick)
    cv2.line(frame, (10, 10), (10, 10+t), hud_color, thick)
    # Top right
    cv2.line(frame, (w-10, 10), (w-10-t, 10), hud_color, thick)
    cv2.line(frame, (w-10, 10), (w-10, 10+t), hud_color, thick)
    # Bottom left
    cv2.line(frame, (10, h-10), (10+t, h-10), hud_color, thick)
    cv2.line(frame, (10, h-10), (10, h-10-t), hud_color, thick)
    # Bottom right
    cv2.line(frame, (w-10, h-10), (w-10-t, h-10), hud_color, thick)
    cv2.line(frame, (w-10, h-10), (w-10, h-10-t), hud_color, thick)
    
    # Text data
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Left side: SYS INFO
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 35), font, 0.7, (255, 255, 255), 2)
    avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
    cv2.putText(frame, f"CONF: {avg_conf:.2f}", (30, 65), font, 0.7, (255, 255, 255), 2)
    
    # Right side: ANALYSIS
    cv2.putText(frame, f"FLOOD AREA: {flood_percentage:.1f}%", (w - 300, 35), font, 0.7, hud_color, 2)
    cv2.putText(frame, f"SEVERITY: {severity}", (w - 300, 65), font, 0.7, hud_color, 2)
    
    return frame

def apply_glowing_mask(frame, mask, is_dangerous=False):
    """Apply an additive glowing semi-transparent mask."""
    # Color: Red for dangerous, Cyan for normal flood
    color = (0, 0, 255) if is_dangerous else (255, 200, 0) # BGR
    
    colored_mask = np.zeros_like(frame)
    colored_mask[mask > 0] = color
    
    # Add strong blur for the glow
    blurred_mask = cv2.GaussianBlur(colored_mask, (41, 41), 0)
    
    # Mix flat mask and blurred mask for the final overlay
    overlay = cv2.addWeighted(colored_mask, 0.5, blurred_mask, 0.8, 0)
    
    # Additive blend with the original frame
    result = cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)
    return result

def run_advanced_inference(model_path, source, save_path=None, conf_thres=0.25):
    print(f"[INFO] Loading YOLOv8 model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found at {model_path}. Please check the path.")
        return
        
    model = YOLO(model_path)
    
    print(f"[INFO] Opening source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {source}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps == 0 or np.isnan(cam_fps):
        cam_fps = 30.0
        
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, cam_fps, (width, height))
        print(f"[INFO] Saving output to {save_path}")

    print("[INFO] Starting inference. Press 'q' to quit.")
    
    # Used for FPS smoothing
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        curr_time = time.time()
        
        # Run YOLOv8 inference
        results = model.predict(frame, conf=conf_thres, verbose=False)
        result = results[0]
        
        flood_percentage = 0.0
        severity = "NORMAL"
        confidences = []
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        if result.masks is not None and result.boxes is not None:
            # Get masks and boxes
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            
            for i, mask in enumerate(masks):
                # Resize mask to frame dimensions
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.logical_or(combined_mask, mask_resized > 0.5).astype(np.uint8)
                confidences.append(boxes.conf[i].item())
                
            # Calculate flood percentage
            total_pixels = width * height
            flood_pixels = np.sum(combined_mask)
            flood_percentage = (flood_pixels / total_pixels) * 100
            
            # Determine severity
            is_dangerous = False
            if flood_percentage > 25.0:
                severity = "CRITICAL"
                is_dangerous = True
            elif flood_percentage > 10.0:
                severity = "WARNING"
                
            # Apply glowing mask effect
            frame = apply_glowing_mask(frame, combined_mask, is_dangerous=is_dangerous)
            
            # Draw bounding boxes
            for box, conf in zip(boxes.xyxy, confidences):
                x1, y1, x2, y2 = map(int, box[:4])
                color = (0, 0, 255) if is_dangerous else (255, 255, 0)
                
                # Draw minimal futuristic brackets for boxes instead of full rectangle
                length = 15
                cv2.line(frame, (x1, y1), (x1+length, y1), color, 2)
                cv2.line(frame, (x1, y1), (x1, y1+length), color, 2)
                cv2.line(frame, (x2, y2), (x2-length, y2), color, 2)
                cv2.line(frame, (x2, y2), (x2, y2-length), color, 2)
                
        # Calculate smoothed FPS
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Draw HUD
        frame = draw_futuristic_ui(frame, fps, flood_percentage, severity, confidences)
        
        # Write to video
        if writer:
            writer.write(frame)
            
        # Display
        try:
            cv2.imshow("FloodSense AI - Tactical Dashboard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            if not getattr(run_advanced_inference, "_logged_gui_warning", False):
                print(f"[WARNING] Could not open live display window: {e}. Video processing and output saving will continue in background.")
                run_advanced_inference._logged_gui_warning = True

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Futuristic Flood Detection System")
    parser.add_argument("--model", type=str, default="runs/segment/train/weights/best.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--source", type=str, default="0", help="0 for webcam, or path to video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Detect webcam vs file
    source = int(args.source) if args.source.isdigit() else args.source
    
    run_advanced_inference(args.model, source, args.output, args.conf)
