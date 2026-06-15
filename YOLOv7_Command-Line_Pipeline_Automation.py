import os
import gc
import sys
import shutil
import glob
import time
import subprocess
import pandas as pd
import torch

# ==============================================================================
# 1. USER CONFIGURATION BLOCK (NOTIFICATION: ADJUST PATHS & ARCHITECTURE HERE)
# ==============================================================================
# TODO: Update these base environments to reflect your local folder layout
WORKING_BASE_DIR = r"C:\Path\To\Your\Main\Workspace"
YOLOV7_REPOSITORY = os.path.join(WORKING_BASE_DIR, "yolov7")  # Root location of cloned YOLOv7 source code

# --- Pipeline Constants ---
RUN_NAME = "yolov7_custom_training_run"
EPOCHS = 50
BATCH_SIZE = 8
IMAGE_SIZE = 640

# Absolute Path Absolute Path Converters
TRAIN_SCRIPT = os.path.join(YOLOV7_REPOSITORY, "train.py")
WEIGHTS_PATH = os.path.join(YOLOV7_REPOSITORY, "yolov7_training.pt")
CONFIG_PATH = os.path.join(YOLOV7_REPOSITORY, "cfg", "training", "yolov7.yaml")
HYP_PATH = os.path.join(YOLOV7_REPOSITORY, "data", "hyp.scratch.p5.yaml")
DATA_YAML_PATH = os.path.join(WORKING_BASE_DIR, "Dataset_Folder", "data.yaml")
OUTPUT_PROJECT_DIR = os.path.join(WORKING_BASE_DIR, "OUTPUT_MODELS")


# ==============================================================================
# 2. RUNTIME WORKSPACE PURGING UTILITIES
# ==============================================================================
def empty_hardware_vram():
    """Flushes active PyTorch CUDA execution buffers and drops stale context weights."""
    print("\n[1/3] Flushing active GPU hardware memory allocations...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[SUCCESS] Hardware CUDA VRAM cache cleared.")
    else:
        print("[INFO] CUDA environment missing or unattached to the runtime execution thread.")
    gc.collect()


def drop_stale_caches(dataset_path):
    """Purges system caching logs from input images to force clean verification mapping."""
    print("[2/3] Scrubbing existing dataset label verification cache targets...")
    cache_files = glob.glob(os.path.join(dataset_path, "**", "*.cache"), recursive=True)
    cleared = 0
    for file_path in cache_files:
        try:
            os.remove(file_path)
            cleared += 1
        except Exception:
            pass
    print(f"[SUCCESS] System cleaned. Dropped {cleared} temporary verification files from scope.")


def preparation_output_workspace(project_path, current_run_name):
    """Ensures structural workspace cleanliness by dropping existing runs with conflicting tags."""
    print("[3/3] Auditing destination file structures...")
    target_dir = os.path.join(project_path, current_run_name)
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            print(f"[SUCCESS] Purged stale conflicting workspace directory trees from: {target_dir}")
        except Exception as e:
            print(f"[WARNING] Local system access limitation encountered cleaning workspace directories: {e}")
    else:
        print("[INFO] Target directory validation complete. Workspace clean.")


# ==============================================================================
# 3. TEXT PERFORMANCE LOG ANALYSIS CALCULATOR
# ==============================================================================
def process_logs_to_excel(project_dir, name, execution_intervals):
    """Translates the traditional results.txt log values into a high-density analytics report."""
    results_txt = os.path.join(project_dir, name, "results.txt")
    output_xlsx = os.path.join(project_dir, name, "yolov7_matrix_metrics.xlsx")
    
    if not os.path.exists(results_txt):
        print(f"[WARNING] The raw tracking text matrix file could not be mapped: {results_txt}")
        return
        
    try:
        parsed_data = []
        with open(results_txt, 'r') as log_stream:
            for log_line in log_stream:
                segments = log_line.strip().split()
                if segments:
                    parsed_data.append(segments)
                    
        if not parsed_data:
            print("[WARNING] Extraction scan returned completely blank metrics layers.")
            return
            
        df = pd.DataFrame(parsed_data)
        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column], errors='raise')
            except Exception:
                pass
                
        base_headers = [
            "Epoch", "GPU_Mem", "Train_Box_Loss", "Train_Obj_Loss", "Train_Cls_Loss", 
            "Total_Loss", "Labels", "Img_Size", "Precision", "Recall", "mAP50", "mAP50-95"
        ]
        
        if len(df.columns) >= len(base_headers):
            df.columns = base_headers + [f"Col_{idx}" for idx in range(len(base_headers), len(df.columns))]
            
        if df["Epoch"].dtype == object:
            df["Epoch"] = df["Epoch"].astype(str).str.split('/').str[0]
        df["Epoch"] = pd.to_numeric(df["Epoch"], errors='coerce').fillna(0) + 1
        
        if "mAP50" in df.columns and pd.api.types.is_numeric_dtype(df["mAP50"]):
            df["% Accuracy (Box)"] = df["mAP50"] * 100
            df["% Error (Box)"] = 100.0 - df["% Accuracy (Box)"]
            
        if len(execution_intervals) >= len(df):
            execution_intervals = execution_intervals[:len(df)]
        else:
            execution_intervals += [0] * (len(df) - len(execution_intervals))
            
        df["Epoch_Duration (sec)"] = [round(t, 1) for t in execution_intervals]
        df["Epoch_Duration (mm:ss)"] = [f"{int(t // 60)}m {int(t % 60)}s" for t in execution_intervals]
        
        df.to_excel(output_xlsx, index=False)
        print(f"[SUCCESS] Compiled performance reports cleanly exported to:\n{output_xlsx}")
    except Exception as err:
        print(f"[ERROR] Structured log conversion process encountered an unexpected failure: {err}")


# ==============================================================================
# 4. SUBSYSTEM PROCESS ENGINE INITIATION
# ==============================================================================
def run_pipeline():
    empty_hardware_vram()
    drop_stale_caches(os.path.dirname(DATA_YAML_PATH))
    preparation_output_workspace(OUTPUT_PROJECT_DIR, RUN_NAME)
    
    # Structural command configuration mapping for native execution context
    cmd = [
        "python", TRAIN_SCRIPT,
        "--workers", "2",
        "--cache-images",
        "--weights", WEIGHTS_PATH,
        "--cfg", CONFIG_PATH,
        "--data", DATA_YAML_PATH,
        "--hyp", HYP_PATH,
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--img-size", str([IMAGE_SIZE, IMAGE_SIZE])[1:-1].replace(" ", ""),
        "--project", OUTPUT_PROJECT_DIR,
        "--name", RUN_NAME,
        "--device", "0"
    ]
    
    print("\n" + "="*60)
    print(" INITIATING SUBSYSTEM TERMINAL EXECUTION PIPELINE FOR YOLOV7 ")
    print("="*60)
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        text=True, encoding='utf-8', errors='ignore'
    )
    
    epoch_times = []
    epoch_start_time = None
    
    while True:
        output_line = process.stdout.readline()
        if not output_line and process.poll() is not None:
            break
        if output_line:
            sys.stdout.write(output_line)
            sys.stdout.flush()
            
            if "Starting training for epoch" in output_line:
                epoch_start_time = time.time()
                
            parts = output_line.strip().split()
            if parts and parts[0] == "all" and len(parts) >= 6 and epoch_start_time is not None:
                duration = time.time() - epoch_start_time
                epoch_times.append(duration)
                epoch_start_time = time.time()  # Reset step markers
                
    process.wait()
    
    if process.returncode == 0:
        process_logs_to_excel(OUTPUT_PROJECT_DIR, RUN_NAME, epoch_times)
    else:
        print("\n[ERROR] The underlying YOLOv7 script environment optimization terminated with a failure exception.")

if __name__ == "__main__":
    run_pipeline()