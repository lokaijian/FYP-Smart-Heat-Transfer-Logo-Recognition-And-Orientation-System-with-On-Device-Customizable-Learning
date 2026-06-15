import os
import pandas as pd
import torch
from ultralytics import YOLO

# ==============================================================================
# 1. USER CONFIGURATION BLOCK (NOTIFICATION: ADJUST PATHS & ARCHITECTURE HERE)
# ==============================================================================
# TODO: Define your local environment paths before running the script
DATASET_DIR = r"C:\Path\To\Your\Dataset"      # Must contain your 'data.yaml' file
PROJECT_DIR = r"C:\Path\To\Your\Output"       # Root directory to save training logs and weights

# TODO: Select your target model architecture by uncommenting your choice
MODEL_SELECTION = 'yolov5nu.pt'        # Option A: YOLOv5 Nano
# MODEL_SELECTION = 'yolov8n.pt'         # Option B: YOLOv8 Nano
# MODEL_SELECTION = 'yolov8s.pt'         # Option C: YOLOv8 Small
# MODEL_SELECTION = 'yolov8m.pt'         # Option D: YOLOv8 Medium
# MODEL_SELECTION = 'yolov8m-pose.pt'    # Option E: YOLOv8 Medium Pose Estimation

# --- Core Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 16  # Note: Reduce to 8 or 4 if your GPU encounters VRAM Out-Of-Memory errors
IMAGE_SIZE = 640

# Auto-generate descriptive run names based on selections
run_name = f"{MODEL_SELECTION.split('.')[0]}_training_run"
full_run_path = os.path.join(PROJECT_DIR, run_name)


# ==============================================================================
# 2. RUNTIME TRACKING & HARDWARE CALIBRATION
# ==============================================================================
gpu_memory_log = []

def track_gpu_mem(trainer):
    """Callback function triggered at each epoch boundary to monitor hardware telemetry."""
    if torch.cuda.is_available():
        # Converts standard allocation bytes directly into Gigabytes (GB)
        mem_gb = torch.cuda.memory_reserved() / 1e9
        gpu_memory_log.append(round(mem_gb, 3))
    else:
        gpu_memory_log.append(0.0)


def main():
    print(f"[INFO] Initializing Core Architecture: {MODEL_SELECTION}")
    model = YOLO(MODEL_SELECTION)
    
    # Register telemetry callback tracking directly into the active training loops
    model.add_callback("on_train_epoch_end", track_gpu_mem)
    
    print(f"[INFO] Launching deep learning optimization loop for: {run_name}")
    model.train(
        data=os.path.join(DATASET_DIR, 'data.yaml'),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=run_name,
        plots=True
    )
    
    # ==============================================================================
    # 3. POST-TRAINING METRIC MATRIX COMPILATION
    # ==============================================================================
    print("\n[INFO] Training concluded. Extracting evaluation metrics into structured analytical sheets...")
    csv_path = os.path.join(full_run_path, 'results.csv')
    excel_path = os.path.join(full_run_path, f'{run_name}_metrics_analysis.xlsx')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # Clean leading/trailing spaces from CSV headers
        
        analysis_df = pd.DataFrame()
        analysis_df['Epoch'] = df['epoch']
        
        # Failsafe protection against telemetry truncation if training was aborted early
        if len(gpu_memory_log) == len(df):
            analysis_df['GPU Memory (GB)'] = gpu_memory_log
        else:
            analysis_df['GPU Memory (GB)'] = gpu_memory_log[:len(df)] + [0.0] * (len(df) - len(gpu_memory_log))
            
        # Parse available bounding box performance statistics safely
        analysis_df['Precision (Box)'] = df.get('metrics/precision(B)', df.get('metrics/precision (B)', 0))
        analysis_df['Recall (Box)'] = df.get('metrics/recall(B)', 0)
        analysis_df['mAP50 (Box)'] = df.get('metrics/mAP50(B)', 0)
        analysis_df['mAP50-95 (Box)'] = df.get('metrics/mAP50-95(B)', 0)
        
        # Pull structural loss values
        analysis_df['Train Box Loss'] = df.get('train/box_loss', 0)
        analysis_df['Train Class Loss'] = df.get('train/cls_loss', 0)
        analysis_df['Train DFL Loss'] = df.get('train/dfl_loss', 0)
        analysis_df['Val Box Loss'] = df.get('val/box_loss', 0)
        analysis_df['Val Class Loss'] = df.get('val/cls_loss', 0)
        analysis_df['Val DFL Loss'] = df.get('val/dfl_loss', 0)
        
        # Convert baseline map indices into explicit evaluation metrics
        if 'metrics/mAP50(B)' in df.columns:
            analysis_df['% Accuracy (Box)'] = df['metrics/mAP50(B)'] * 100
            analysis_df['% Error (Box)'] = (1.0 - df['metrics/mAP50(B)']) * 100
            
        analysis_df.to_excel(excel_path, index=False)
        print(f"[SUCCESS] High-density analytical metrics exported to:\n{excel_path}")
    else:
        print(f"[ERROR] Expected results source data missing at destination: {csv_path}")

if __name__ == '__main__':
    main()