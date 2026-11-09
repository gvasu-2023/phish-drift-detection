"""Master script to run the entire pipeline end-to-end."""
import subprocess
import sys
import time

scripts = [
    "scripts/build_temporal_dataset.py",
    "scripts/train_and_save_model.py"
    "scripts/run_temporal_eval.py",
    "scripts/run_shap_audit.py",
    "scripts/run_prior_adaptation.py",
    "scripts/run_model_comparison.py",
    "scripts/run_calibration_ablation.py",
    "scripts/run_adversarial_eval.py",
    "scripts/run_adversarial_training.py",
    "scripts/run_multi_year_drift.py",
    "scripts/run_cross_dataset_eval.py",
    "scripts/run_phiusiil_eval.py",
    "scripts/run_combined_training.py",
    "scripts/run_drift_detection.py"
]

def run_script(script):
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}\n")
    start_time = time.time()
    
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script} failed. Stopping pipeline.")
        sys.exit(1)
        
    print(f"\n✅ Finished {script} in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    print("Starting Phish-Drift Detection Pipeline...")
    for script in scripts:
        run_script(script)
    print("\n🎉 ALL SCRIPTS COMPLETED SUCCESSFULLY! 🎉")