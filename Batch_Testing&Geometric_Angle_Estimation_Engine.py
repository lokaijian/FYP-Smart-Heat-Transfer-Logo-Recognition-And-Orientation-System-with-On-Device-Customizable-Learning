import os
import math
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. USER CONFIGURATION BLOCK (NOTIFICATION: ADJUST PATHS & TARGET MODEL HERE)
# ==============================================================================
# TODO: Provide localized folder directories to point to your evaluation validation datasets
INPUT_DIR = r"C:\Path\To\Your\Raw_Test_Images"
OUTPUT_DIR = r"C:\Path\To\Your\Inference_Results"
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "batch_evaluation_telemetry.csv")

# TODO: Specify the absolute file path pointing to your compiled weights
MODEL_PATH = r"C:\Path\To\Your\Output\Weights\best.pt"


def setup_directories():
    """Initializes destination folders safely if they are missing on local storage blocks."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created inference generation directory destination path at: {OUTPUT_DIR}")


def draw_geometric_angle(frame, pt1, pt2, current_angle):
    """Draws spatial keypoint coordinates, baseline systems, and descriptive orientation arcs."""
    # Enforce strict translation into absolute integer coordinate pairs for OpenCV operations
    p1 = (int(pt1[0]), int(pt1[1]))
    p2 = (int(pt2[0]), int(pt2[1]))
    
    # 1. Node Render (Blue Node for the Pivot Origin point, Magenta Node for the direction indicator)
    cv2.circle(frame, p1, 8, (255, 0, 0), -1)
    cv2.circle(frame, p2, 8, (255, 0, 255), -1)
    
    # 2. Structural Line Render connecting structural keypoints
    cv2.line(frame, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)
    
    # 3. Horizontal 0-Degree Horizontal Baseline Reference Grid Matrix Vector
    ref_length = 80
    p_ref = (p1[0] + ref_length, p1[1])
    cv2.line(frame, p1, p_ref, (200, 200, 200), 2, cv2.LINE_AA)
    
    # Reference system text label mapping
    cv2.putText(frame, "0 deg Ref", (p_ref[0] + 5, p_ref[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    
    # 4. Generate the Curved Orientation Overlay Arc (Using structural ellipse calculations)
    arc_radius = 40
    if current_angle != 0:
        cv2.ellipse(frame, p1, (arc_radius, arc_radius), 0, 0, int(current_angle),
                    (0, 165, 255), 3, cv2.LINE_AA)


def calculate_angles(kpts):
    """Extracts spatial coordinate data layers to compute rotation adjustments."""
    xy = kpts.xy[0].cpu().numpy()
    
    # Validate coordinate completeness across bounding keypoint pairs
    if len(xy) >= 2 and xy[0][0] != 0 and xy[1][0] != 0:
        pt1 = xy[0]
        pt2 = xy[1]
        
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        
        # Calculate angle orientation values in standard degrees
        current_angle = math.degrees(math.atan2(dy, dx))
        rotation_needed = -current_angle
        
        return round(current_angle, 1), round(rotation_needed, 1), pt1, pt2
    return None, None, None, None


def run_batch_test():
    setup_directories()
    print(f"[INFO] Initializing structural testing context using model weights: {MODEL_PATH}")
    
    try:
        # Enforce target task resolution directly into model declarations
        model = YOLO(MODEL_PATH, task='pose')
    except Exception as err:
        print(f"[ERROR] Failed to properly construct specified architecture model weights context: {err}")
        return
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"[WARNING] Image extraction scan evaluated empty at target directory path location: {INPUT_DIR}")
        return
        
    print(f"[INFO] Located {len(image_files)} test images. Commencing evaluation tracking sequences...\n")
    
    with open(LOG_FILE_PATH, mode='w', encoding='utf-8') as log_file:
        # Initialize output CSV header metrics logs
        log_file.write("Image_Name,Detection_Class,Current_Angle_Deg,Rotation_Needed_Deg\n")
        
        for idx, img_name in enumerate(image_files, start=1):
            img_path = os.path.join(INPUT_DIR, img_name)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"[{idx}] Subsystem read operation exception triggered on target file: {img_name}")
                continue
                
            # Execute validation parsing matrices using standardized threshold barriers
            results = model(frame, conf=0.6, verbose=False)
            
            detected_class = "None"
            curr_ang_str = "N/A"
            rot_need_str = "N/A"
            
            for result in results:
                if len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names[cls_id]
                        detected_class = cls_name
                        
                        # Gather bounding framework anchors for descriptive bounding boxes
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        box_color = (0, 255, 0) if cls_name == 'logo' else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                        
                        if cls_name == 'blank':
                            # Apply alert masks for blank classifications
                            cv2.putText(frame, "STATUS: BLANK DETECTED", (40, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4, cv2.LINE_AA)
                                        
                        elif cls_name == 'logo':
                            if result.keypoints is not None:
                                current_angle, rotation_needed, pt1, pt2 = calculate_angles(result.keypoints[i])
                                
                                if current_angle is not None:
                                    curr_ang_str = str(current_angle)
                                    rot_need_str = str(rotation_needed)
                                    
                                    # Render orientation overlays
                                    draw_geometric_angle(frame, pt1, pt2, current_angle)
                                    
                                    # Append text overlay readouts to image array data structures
                                    cv2.putText(frame, f"Current Angle: {current_angle} deg", (40, 60), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)
                                    cv2.putText(frame, f"Rotate Needed: {rotation_needed} deg", (40, 110), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
                        
                        # Render descriptive bounding box header label tags
                        cv2.putText(frame, cls_name.upper(), (x1, y1 - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2, cv2.LINE_AA)
            
            # Save the annotated frames back into destination folders
            output_img_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
            cv2.imwrite(output_img_path, frame)
            
            # Write row metrics back into spreadsheet file logs
            log_file.write(f"{img_name},{detected_class},{curr_ang_str},{rot_need_str}\n")
            print(f"[{idx}/{len(image_files)}] Compiled: {img_name} -> Category: {detected_class} | Angle: {curr_ang_str}° | Turn: {rot_need_str}°")
            
    print(f"\n[SUCCESS] Verification pipeline finished execution loop maps.")
    print(f" -> Processed Images Output Path: {OUTPUT_DIR}")
    print(f" -> Telemetry Matrix Log Saved To: {LOG_FILE_PATH}")

if __name__ == "__main__":
    run_test_suite = run_batch_test()