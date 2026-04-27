"""One-shot training script — trains all models in sequence."""
import subprocess, sys

def run(script):
    print(f"\n{'='*60}\nRunning: {script}\n{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR in {script}")
        sys.exit(1)

run("data/generate_data.py")
run("churn_engine/churn_model.py")
run("nba_engine/nba_model.py")
print("\n✅ All models trained. Run the dashboard: streamlit run dashboard/app.py")
