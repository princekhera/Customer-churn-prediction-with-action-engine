"""
One-click training script for all Marketing AI models.
Run this after generating data.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("  Marketing AI Platform — Model Training")
    print("=" * 60)
    
    # Step 1: Generate data
    if not os.path.exists("data/master_dataset.csv"):
        print("\n[1/3] Generating synthetic customer data...")
        from data.generate_data import build_master_dataset
        import pandas as pd
        
        master, customers, behavioral, transactional, campaigns = build_master_dataset()
        os.makedirs("data", exist_ok=True)
        master.to_csv("data/master_dataset.csv", index=False)
        customers.to_csv("data/customers.csv", index=False)
        behavioral.to_csv("data/behavioral.csv", index=False)
        transactional.to_csv("data/transactional.csv", index=False)
        campaigns.to_csv("data/campaigns.csv", index=False)
        print(f"   ✅ Generated {len(master):,} customer records")
    else:
        print("\n[1/3] Data already exists — skipping generation")
        import pandas as pd
        master = pd.read_csv("data/master_dataset.csv")
        print(f"   📂 Loaded {len(master):,} records")
    
    # Step 2: Train churn model
    print("\n[2/3] Training Churn Prediction Model...")
    from models.churn_model import ChurnModel
    
    churn = ChurnModel("models")
    metrics = churn.train(master)
    
    print(f"   ✅ Churn Model:")
    print(f"      AUC-ROC:        {metrics['auc_roc']:.4f}")
    print(f"      Avg Precision:  {metrics['avg_precision']:.4f}")
    print(f"      Churn Rate:     {metrics['churn_rate']:.1%}")
    
    # Step 3: Train NBA model
    print("\n[3/3] Training Next-Best-Action Model...")
    from models.nba_model import NextBestActionModel
    
    nba = NextBestActionModel("models")
    nba_metrics = nba.train(master)
    
    print(f"   ✅ NBA Model:")
    print(f"      Accuracy:       {nba_metrics['accuracy']:.4f}")
    print(f"      Features:       {nba_metrics['n_features']}")
    print(f"      Bandit Reward:  {nba_metrics['bandit_avg_reward']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print("\nFiles saved:")
    print("  data/master_dataset.csv")
    print("  models/xgb_churn.pkl")
    print("  models/lgb_churn.pkl")
    print("  models/nba_classifier.pkl")
    print("  models/bandit.pkl")
    print("  models/churn_meta.json")
    print("  models/nba_meta.json")
    
    print("\n" + "=" * 60)
    print("  🚀 NEXT STEPS")
    print("=" * 60)
    print("\n  Run the Streamlit Dashboard:")
    print("    streamlit run dashboard/app.py")
    print("\n  Run the FastAPI Backend:")
    print("    uvicorn api.main:app --reload --port 8000")
    print("\n  API Docs:")
    print("    http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()
