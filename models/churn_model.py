"""
Module 1: Customer Churn Prediction + Action Engine
- XGBoost/LightGBM ensemble churn model
- SHAP explanations
- Rule-based + ML action recommendation engine
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

import xgboost as xgb
import lightgbm as lgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed — explanations will be limited")


# ─── Feature Engineering ──────────────────────────────────────────────────────

CATEGORICAL_COLS = ["segment", "region", "channel", "spend_trend"]
FEATURE_COLS = [
    "age", "tenure_months",
    "logins_last_30d", "avg_session_min", "feature_usage_score",
    "support_tickets_90d", "days_since_last_login", "mobile_app_opens_30d",
    "monthly_spend", "purchase_frequency", "avg_order_value",
    "total_lifetime_value", "last_purchase_days_ago", "discount_usage_rate",
    "payment_failures",
    "total_campaigns_received", "email_open_rate", "click_through_rate",
    "conversion_rate", "total_campaign_revenue",
    # Encoded categoricals appended below
]

FEATURE_DISPLAY_NAMES = {
    "age": "Customer Age",
    "tenure_months": "Tenure (Months)",
    "logins_last_30d": "Logins Last 30 Days",
    "avg_session_min": "Avg Session Duration (min)",
    "feature_usage_score": "Feature Usage Score",
    "support_tickets_90d": "Support Tickets (90d)",
    "days_since_last_login": "Days Since Last Login",
    "mobile_app_opens_30d": "Mobile App Opens (30d)",
    "monthly_spend": "Monthly Spend ($)",
    "purchase_frequency": "Purchase Frequency",
    "avg_order_value": "Avg Order Value ($)",
    "total_lifetime_value": "Total LTV ($)",
    "last_purchase_days_ago": "Days Since Last Purchase",
    "discount_usage_rate": "Discount Usage Rate",
    "payment_failures": "Payment Failures",
    "total_campaigns_received": "Campaigns Received",
    "email_open_rate": "Email Open Rate",
    "click_through_rate": "Click-Through Rate",
    "conversion_rate": "Campaign Conversion Rate",
    "total_campaign_revenue": "Campaign Revenue ($)",
    "segment_enc": "Customer Segment",
    "region_enc": "Region",
    "channel_enc": "Channel",
    "spend_trend_enc": "Spend Trend",
}


def engineer_features(df):
    """Feature engineering pipeline"""
    df = df.copy()
    
    # Encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Derived features
    if "logins_last_30d" in df.columns and "tenure_months" in df.columns:
        df["engagement_score"] = (
            df["logins_last_30d"] * 0.4 +
            df["feature_usage_score"] * 0.3 +
            df["mobile_app_opens_30d"] * 0.3
        ).round(2)
    
    if "monthly_spend" in df.columns and "purchase_frequency" in df.columns:
        df["spend_per_login"] = np.where(
            df["logins_last_30d"] > 0,
            df["monthly_spend"] / df["logins_last_30d"],
            0
        ).round(2)
    
    if "days_since_last_login" in df.columns:
        df["recency_score"] = np.exp(-df["days_since_last_login"] / 30).round(3)
    
    return df, label_encoders


def get_feature_columns(df):
    base = [c for c in FEATURE_COLS if c in df.columns]
    enc = [c for c in df.columns if c.endswith("_enc")]
    derived = ["engagement_score", "spend_per_login", "recency_score"]
    derived = [c for c in derived if c in df.columns]
    return list(dict.fromkeys(base + enc + derived))


# ─── Model Training ───────────────────────────────────────────────────────────

class ChurnModel:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.xgb_model = None
        self.lgb_model = None
        self.feature_cols = None
        self.label_encoders = {}
        self.shap_explainer = None
        self.metrics = {}
    
    def train(self, df):
        print("🔧 Engineering features...")
        df_eng, self.label_encoders = engineer_features(df)
        self.feature_cols = get_feature_columns(df_eng)
        
        X = df_eng[self.feature_cols].fillna(0)
        y = df_eng["churn_label"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training: {len(X_train)} samples | Test: {len(X_test)} samples")
        print(f"   Churn rate: {y.mean():.1%}")
        
        # XGBoost
        print("\n🚀 Training XGBoost...")
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # LightGBM
        print("🚀 Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.lgb_model.fit(X_train, y_train)
        
        # Ensemble predictions (average probabilities)
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X_test)[:, 1]
        ensemble_prob = (xgb_prob * 0.5 + lgb_prob * 0.5)
        
        preds = (ensemble_prob >= 0.5).astype(int)
        
        self.metrics = {
            "auc_roc": round(roc_auc_score(y_test, ensemble_prob), 4),
            "avg_precision": round(average_precision_score(y_test, ensemble_prob), 4),
            "xgb_auc": round(roc_auc_score(y_test, xgb_prob), 4),
            "lgb_auc": round(roc_auc_score(y_test, lgb_prob), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "churn_rate": round(float(y.mean()), 4),
        }
        
        print(f"\n📊 Model Metrics:")
        print(f"   Ensemble AUC-ROC:  {self.metrics['auc_roc']:.4f}")
        print(f"   Avg Precision:     {self.metrics['avg_precision']:.4f}")
        print(f"   XGBoost AUC:       {self.metrics['xgb_auc']:.4f}")
        print(f"   LightGBM AUC:      {self.metrics['lgb_auc']:.4f}")
        
        # SHAP explainer
        if SHAP_AVAILABLE:
            print("\n🔍 Building SHAP explainer...")
            self.shap_explainer = shap.TreeExplainer(self.xgb_model)
            shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
            self.shap_values = self.shap_explainer.shap_values(shap_sample)
            self.shap_sample = shap_sample
        
        self.save()
        print(f"\n✅ Model saved to {self.model_dir}/")
        return self.metrics
    
    def predict(self, df):
        """Predict churn probability for a dataframe"""
        df_eng, _ = engineer_features(df)
        
        # Re-apply stored encoders
        for col, le in self.label_encoders.items():
            if col in df_eng.columns:
                df_eng[f"{col}_enc"] = df_eng[col].map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        X = df_eng[self.feature_cols].fillna(0)
        
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X)[:, 1]
        ensemble_prob = (xgb_prob * 0.5 + lgb_prob * 0.5)
        
        return ensemble_prob
    
    def explain(self, customer_row):
        """Get SHAP explanation for a single customer"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return self._fallback_explanation(customer_row)
        
        df_eng, _ = engineer_features(pd.DataFrame([customer_row]))
        X = df_eng[self.feature_cols].fillna(0)
        
        shap_vals = self.shap_explainer.shap_values(X)
        
        explanation = []
        for i, col in enumerate(self.feature_cols):
            explanation.append({
                "feature": col,
                "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
                "value": float(X.iloc[0][col]),
                "shap_value": float(shap_vals[0][i]),
            })
        
        explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return explanation[:10]
    
    def _fallback_explanation(self, customer_row):
        """Simple feature importance when SHAP unavailable"""
        if self.xgb_model is None:
            return []
        importances = self.xgb_model.feature_importances_
        explanation = []
        for i, col in enumerate(self.feature_cols):
            explanation.append({
                "feature": col,
                "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
                "value": float(customer_row.get(col, 0)),
                "shap_value": float(importances[i]),
            })
        explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return explanation[:10]
    
    def save(self):
        joblib.dump(self.xgb_model, f"{self.model_dir}/xgb_churn.pkl")
        joblib.dump(self.lgb_model, f"{self.model_dir}/lgb_churn.pkl")
        joblib.dump(self.label_encoders, f"{self.model_dir}/label_encoders.pkl")
        
        meta = {
            "feature_cols": self.feature_cols,
            "metrics": self.metrics,
        }
        with open(f"{self.model_dir}/churn_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    def load(self):
        self.xgb_model = joblib.load(f"{self.model_dir}/xgb_churn.pkl")
        self.lgb_model = joblib.load(f"{self.model_dir}/lgb_churn.pkl")
        self.label_encoders = joblib.load(f"{self.model_dir}/label_encoders.pkl")
        
        with open(f"{self.model_dir}/churn_meta.json") as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        self.metrics = meta["metrics"]
        
        if SHAP_AVAILABLE:
            self.shap_explainer = shap.TreeExplainer(self.xgb_model)
        return self


# ─── Action Engine ────────────────────────────────────────────────────────────

class ActionEngine:
    """
    Next-Best-Action engine for churn prevention.
    Determines the optimal intervention based on churn probability + customer profile.
    """
    
    ACTIONS = {
        "aggressive_discount": {
            "label": "Aggressive Discount (20–30%)",
            "description": "High-value customer at risk. Offer substantial discount to retain.",
            "icon": "💰",
            "priority": 1,
            "cost": "high",
        },
        "personal_call": {
            "label": "Personal Success Call",
            "description": "Schedule 1:1 call with customer success manager.",
            "icon": "📞",
            "priority": 1,
            "cost": "high",
        },
        "winback_email": {
            "label": "Win-Back Email Campaign",
            "description": "Personalized email highlighting missed features and value.",
            "icon": "📧",
            "priority": 2,
            "cost": "low",
        },
        "loyalty_reward": {
            "label": "Loyalty Reward",
            "description": "Award bonus points or credits to re-engage.",
            "icon": "🎁",
            "priority": 2,
            "cost": "medium",
        },
        "feature_nudge": {
            "label": "Feature Education Push",
            "description": "In-app nudge highlighting underused features.",
            "icon": "💡",
            "priority": 3,
            "cost": "low",
        },
        "light_email": {
            "label": "Engagement Newsletter",
            "description": "Light newsletter with tips and success stories.",
            "icon": "📰",
            "priority": 3,
            "cost": "low",
        },
        "do_nothing": {
            "label": "Monitor Only",
            "description": "Low risk — monitor and re-evaluate next month.",
            "icon": "👀",
            "priority": 4,
            "cost": "none",
        },
    }
    
    def recommend(self, customer: dict, churn_prob: float, explanation: list = None) -> dict:
        """
        Recommend actions for a customer based on churn probability and profile.
        Returns ranked list of recommended actions with reasoning.
        """
        actions = []
        reasoning = []
        
        segment = customer.get("segment", "Regular")
        ltv = float(customer.get("total_lifetime_value", 0))
        support_tickets = int(customer.get("support_tickets_90d", 0))
        days_inactive = int(customer.get("days_since_last_login", 0))
        feature_usage = float(customer.get("feature_usage_score", 50))
        spend_trend = customer.get("spend_trend", "stable")
        email_open_rate = float(customer.get("email_open_rate", 0.3))
        conversion_rate = float(customer.get("conversion_rate", 0.05))
        
        # ── HIGH RISK: churn > 0.7 ──
        if churn_prob >= 0.7:
            reasoning.append(f"⚠️ HIGH CHURN RISK ({churn_prob:.0%}) — immediate intervention required")
            
            if segment == "Premium" or ltv > 5000:
                actions.append("personal_call")
                reasoning.append(f"Premium/high-LTV customer (${ltv:,.0f}) → Personal call recommended")
                actions.append("aggressive_discount")
            else:
                actions.append("aggressive_discount")
                reasoning.append("Discount offer to break churn momentum")
                actions.append("winback_email")
            
            if support_tickets >= 2:
                reasoning.append(f"{support_tickets} recent support tickets → friction is a driver")
        
        # ── MEDIUM RISK: 0.4 ≤ churn ≤ 0.7 ──
        elif churn_prob >= 0.4:
            reasoning.append(f"🟡 MEDIUM CHURN RISK ({churn_prob:.0%}) — proactive engagement needed")
            
            if days_inactive > 14:
                actions.append("winback_email")
                reasoning.append(f"Inactive for {days_inactive} days → re-engagement email")
            
            if feature_usage < 40:
                actions.append("feature_nudge")
                reasoning.append(f"Low feature usage ({feature_usage:.0f}/100) → education nudge")
            
            if spend_trend == "decreasing":
                actions.append("loyalty_reward")
                reasoning.append("Declining spend trend → loyalty incentive")
            
            if not actions:
                actions.append("light_email")
        
        # ── LOW RISK: churn < 0.4 ──
        else:
            reasoning.append(f"✅ LOW CHURN RISK ({churn_prob:.0%}) — maintain engagement")
            
            if email_open_rate > 0.4 and conversion_rate > 0.1:
                actions.append("loyalty_reward")
                reasoning.append("Highly engaged customer — loyalty reward can deepen relationship")
            else:
                actions.append("light_email")
                reasoning.append("Regular engagement newsletter to maintain momentum")
            
            actions.append("do_nothing")
        
        # ── Top SHAP drivers ──
        top_drivers = []
        if explanation:
            pos_drivers = [e for e in explanation if e["shap_value"] > 0][:3]
            for d in pos_drivers:
                top_drivers.append(f"{d['display_name']}: {d['value']:.1f}")
        
        # Deduplicate
        seen = set()
        unique_actions = []
        for a in actions:
            if a not in seen:
                unique_actions.append(a)
                seen.add(a)
        
        return {
            "churn_probability": round(churn_prob, 4),
            "risk_tier": (
                "HIGH" if churn_prob >= 0.7 else
                "MEDIUM" if churn_prob >= 0.4 else "LOW"
            ),
            "recommended_actions": [
                {**self.ACTIONS[a], "action_id": a}
                for a in unique_actions if a in self.ACTIONS
            ],
            "reasoning": reasoning,
            "top_churn_drivers": top_drivers,
        }
    
    def bulk_recommend(self, df: pd.DataFrame, churn_probs: np.ndarray) -> pd.DataFrame:
        """Generate action recommendations for a full dataframe"""
        results = []
        for i, row in df.iterrows():
            rec = self.recommend(row.to_dict(), churn_probs[i])
            results.append({
                "customer_id": row.get("customer_id", i),
                "churn_probability": rec["churn_probability"],
                "risk_tier": rec["risk_tier"],
                "primary_action": rec["recommended_actions"][0]["label"] if rec["recommended_actions"] else "Monitor",
                "action_icon": rec["recommended_actions"][0]["icon"] if rec["recommended_actions"] else "👀",
                "n_actions": len(rec["recommended_actions"]),
                "primary_reasoning": rec["reasoning"][0] if rec["reasoning"] else "",
            })
        return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    
    print("Loading data...")
    df = pd.read_csv("data/master_dataset.csv")
    print(f"Dataset: {df.shape}")
    
    model = ChurnModel(model_dir="models")
    metrics = model.train(df)
    
    print("\n--- Testing Action Engine ---")
    engine = ActionEngine()
    
    # Test on a high-risk customer
    high_risk = df[df["churn_label"] == 1].iloc[0].to_dict()
    probs = model.predict(pd.DataFrame([high_risk]))
    explanation = model.explain(high_risk)
    rec = engine.recommend(high_risk, probs[0], explanation)
    
    print(f"\nCustomer: {high_risk['customer_id']} | Segment: {high_risk['segment']}")
    print(f"Churn Prob: {rec['churn_probability']:.0%} | Risk: {rec['risk_tier']}")
    print("Actions:", [a["label"] for a in rec["recommended_actions"]])
    print("Reasoning:", rec["reasoning"][:2])
