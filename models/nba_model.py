"""
Module 2: Next-Best-Action Recommendation System
- Multi-class classification model for action selection
- Contextual Bandit simulation with UCB exploration
- Reward tracking and performance over time
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb


# ─── Action Space ─────────────────────────────────────────────────────────────

ACTIONS = [
    "send_email",
    "push_notification",
    "sms_offer",
    "upsell_offer",
    "loyalty_reward",
    "do_nothing",
]

ACTION_METADATA = {
    "send_email": {
        "label": "Send Email Campaign",
        "icon": "📧",
        "description": "Personalized email with relevant content or offer",
        "base_conversion": 0.12,
        "cost": 0.5,
        "channel": "email",
    },
    "push_notification": {
        "label": "Push Notification",
        "icon": "🔔",
        "description": "In-app or mobile push with time-sensitive message",
        "base_conversion": 0.08,
        "cost": 0.2,
        "channel": "mobile",
    },
    "sms_offer": {
        "label": "SMS Offer",
        "icon": "💬",
        "description": "SMS with exclusive limited-time discount",
        "base_conversion": 0.15,
        "cost": 1.0,
        "channel": "sms",
    },
    "upsell_offer": {
        "label": "Upsell Offer",
        "icon": "⬆️",
        "description": "Targeted upsell based on usage patterns",
        "base_conversion": 0.09,
        "cost": 2.0,
        "channel": "email",
    },
    "loyalty_reward": {
        "label": "Loyalty Reward",
        "icon": "🎁",
        "description": "Award bonus points/credits to drive engagement",
        "base_conversion": 0.18,
        "cost": 1.5,
        "channel": "in-app",
    },
    "do_nothing": {
        "label": "Do Nothing",
        "icon": "⏸️",
        "description": "No action — conserve budget for higher-value moments",
        "base_conversion": 0.02,
        "cost": 0.0,
        "channel": "none",
    },
}

# ─── Label Generation for Optimal Action ─────────────────────────────────────

def compute_optimal_action(df):
    """
    Compute the optimal action label for each customer
    using a heuristic reward function (simulating ground truth)
    """
    labels = []
    for _, row in df.iterrows():
        seg = row.get("segment", "Regular")
        channel = row.get("channel", "Both")
        email_open_rate = float(row.get("email_open_rate", 0.2))
        conv_rate = float(row.get("conversion_rate", 0.05))
        churn_prob = float(row.get("churn_score_true", 0.3))
        ltv = float(row.get("total_lifetime_value", 100))
        days_inactive = int(row.get("days_since_last_login", 5))
        feature_usage = float(row.get("feature_usage_score", 50))
        purchase_freq = int(row.get("purchase_frequency", 3))
        
        # Decision tree of optimal action
        if churn_prob > 0.7 and seg == "Premium":
            label = "sms_offer"
        elif churn_prob > 0.7:
            label = "send_email"
        elif days_inactive > 20:
            if channel in ["Mobile", "Both"]:
                label = "push_notification"
            else:
                label = "send_email"
        elif feature_usage < 30 and purchase_freq > 2:
            label = "upsell_offer"
        elif email_open_rate > 0.4 and conv_rate > 0.1:
            label = "loyalty_reward"
        elif ltv > 3000 and churn_prob < 0.3:
            label = "do_nothing"
        elif channel in ["Mobile", "Both"] and days_inactive < 5:
            label = "push_notification"
        else:
            label = "send_email"
        
        labels.append(label)
    
    return labels


# ─── Contextual Bandit ────────────────────────────────────────────────────────

class ContextualBandit:
    """
    UCB (Upper Confidence Bound) Contextual Bandit
    Balances exploration vs exploitation for action selection
    """
    
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.counts = np.zeros(n_actions)           # Times each arm pulled
        self.values = np.zeros(n_actions)           # Running average reward
        self.total_pulls = 0
        self.history = []
    
    def ucb_score(self, alpha=2.0):
        """Upper Confidence Bound scores"""
        if self.total_pulls == 0:
            return np.ones(self.n_actions) * float("inf")
        
        with np.errstate(divide="ignore", invalid="ignore"):
            exploration = np.where(
                self.counts > 0,
                alpha * np.sqrt(np.log(self.total_pulls) / self.counts),
                float("inf")
            )
        return self.values + exploration
    
    def select_action(self, alpha=2.0):
        scores = self.ucb_score(alpha)
        return int(np.argmax(scores))
    
    def update(self, action_idx, reward):
        self.counts[action_idx] += 1
        self.total_pulls += 1
        n = self.counts[action_idx]
        self.values[action_idx] += (reward - self.values[action_idx]) / n
        self.history.append({
            "pull": self.total_pulls,
            "action": action_idx,
            "reward": reward,
            "avg_reward": float(np.mean(self.values)),
        })
    
    def get_performance(self):
        return pd.DataFrame(self.history) if self.history else pd.DataFrame()


def simulate_reward(action: str, customer: dict) -> float:
    """
    Simulate the reward (conversion) for taking an action on a customer.
    Used for bandit training and evaluation.
    """
    meta = ACTION_METADATA.get(action, {})
    base = meta.get("base_conversion", 0.05)
    
    # Adjust based on customer features
    email_open_rate = float(customer.get("email_open_rate", 0.25))
    churn_prob = float(customer.get("churn_score_true", 0.3))
    ltv = float(customer.get("total_lifetime_value", 200))
    channel = customer.get("channel", "Both")
    purchase_freq = int(customer.get("purchase_frequency", 3))
    
    multiplier = 1.0
    
    if action == "send_email":
        multiplier = email_open_rate / 0.25
    elif action == "push_notification":
        multiplier = 1.5 if channel in ["Mobile", "Both"] else 0.4
    elif action == "sms_offer":
        multiplier = 1.3 if churn_prob > 0.5 else 0.8
    elif action == "upsell_offer":
        multiplier = 1.4 if purchase_freq > 5 else 0.7
    elif action == "loyalty_reward":
        multiplier = 1.2 if ltv > 500 else 0.9
    elif action == "do_nothing":
        multiplier = 0.5 if churn_prob < 0.3 else 0.1
    
    prob = np.clip(base * multiplier + np.random.normal(0, 0.02), 0, 1)
    return float(np.random.random() < prob)


# ─── NBA Model ────────────────────────────────────────────────────────────────

class NextBestActionModel:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.bandits = {a: ContextualBandit(len(ACTIONS)) for a in ACTIONS}
        self.metrics = {}
    
    def _get_feature_cols(self, df):
        numeric_candidates = [
            "age", "tenure_months", "logins_last_30d", "avg_session_min",
            "feature_usage_score", "support_tickets_90d", "days_since_last_login",
            "mobile_app_opens_30d", "monthly_spend", "purchase_frequency",
            "avg_order_value", "total_lifetime_value", "last_purchase_days_ago",
            "discount_usage_rate", "payment_failures", "total_campaigns_received",
            "email_open_rate", "click_through_rate", "conversion_rate",
            "total_campaign_revenue", "churn_score_true",
        ]
        return [c for c in numeric_candidates if c in df.columns]
    
    def _encode_categoricals(self, df):
        df = df.copy()
        cat_cols = ["segment", "region", "channel", "spend_trend"]
        for col in cat_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
        return df
    
    def train(self, df):
        print("🎯 Training Next-Best-Action Model...")
        
        df = self._encode_categoricals(df)
        
        # Generate optimal action labels
        print("   Generating action labels...")
        df["optimal_action"] = compute_optimal_action(df)
        
        self.feature_cols = self._get_feature_cols(df)
        dummy_cols = [c for c in df.columns if c.startswith(("segment_", "region_", "channel_", "spend_trend_"))]
        all_features = self.feature_cols + dummy_cols
        all_features = [c for c in all_features if c in df.columns]
        
        X = df[all_features].fillna(0)
        y = self.label_encoder.fit_transform(df["optimal_action"])
        
        self.feature_cols = all_features
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)
        
        # XGBoost multi-class
        self.classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=len(ACTIONS),
            random_state=42,
            n_jobs=-1,
        )
        self.classifier.fit(X_train_sc, y_train, verbose=False)
        
        y_pred = self.classifier.predict(X_test_sc)
        acc = accuracy_score(y_test, y_pred)
        
        action_dist = df["optimal_action"].value_counts().to_dict()
        
        self.metrics = {
            "accuracy": round(acc, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "action_distribution": action_dist,
            "n_features": len(self.feature_cols),
        }
        
        print(f"   Accuracy: {acc:.3f}")
        print(f"   Action distribution: {action_dist}")
        
        # Simulate bandit rewards
        print("   Simulating bandit rewards (1000 episodes)...")
        sample = df.sample(min(1000, len(df)), random_state=42)
        bandit = ContextualBandit(len(ACTIONS))
        
        for _, row in sample.iterrows():
            action_idx = bandit.select_action()
            action = ACTIONS[action_idx]
            reward = simulate_reward(action, row.to_dict())
            bandit.update(action_idx, reward)
        
        self.global_bandit = bandit
        self.metrics["bandit_avg_reward"] = round(float(np.mean(bandit.values)), 4)
        
        self.save()
        print(f"✅ NBA Model saved.")
        return self.metrics
    
    def recommend(self, customer: dict, top_k: int = 3) -> dict:
        """Recommend top-k actions for a customer"""
        df = pd.DataFrame([customer])
        df = self._encode_categoricals(df)
        
        # Align features
        X = pd.DataFrame(0, index=[0], columns=self.feature_cols)
        for col in self.feature_cols:
            if col in df.columns:
                X[col] = df[col].values[0]
        
        X_sc = self.scaler.transform(X.fillna(0))
        probs = self.classifier.predict_proba(X_sc)[0]
        
        action_labels = self.label_encoder.classes_
        ranked = sorted(
            zip(action_labels, probs),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Simulate expected reward for each top action
        recommendations = []
        for action_label, prob in ranked:
            reward = simulate_reward(action_label, customer)
            meta = ACTION_METADATA.get(action_label, {})
            roi = (reward * 50 - meta.get("cost", 1)) / max(meta.get("cost", 1), 0.01)
            
            recommendations.append({
                "action": action_label,
                "label": meta.get("label", action_label),
                "icon": meta.get("icon", "🎯"),
                "description": meta.get("description", ""),
                "confidence": round(float(prob), 3),
                "channel": meta.get("channel", ""),
                "expected_roi": round(roi, 2),
                "simulated_reward": round(reward, 3),
                "cost": meta.get("cost", 0),
            })
        
        return {
            "customer_id": customer.get("customer_id", "unknown"),
            "segment": customer.get("segment", "Unknown"),
            "recommendations": recommendations,
            "top_action": recommendations[0] if recommendations else None,
        }
    
    def bulk_recommend(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            rec = self.recommend(row.to_dict(), top_k=1)
            top = rec.get("top_action") or {}
            results.append({
                "customer_id": row.get("customer_id", ""),
                "segment": row.get("segment", ""),
                "recommended_action": top.get("label", "Unknown"),
                "action_icon": top.get("icon", ""),
                "confidence": top.get("confidence", 0),
                "expected_roi": top.get("expected_roi", 0),
                "channel": top.get("channel", ""),
            })
        return pd.DataFrame(results)
    
    def save(self):
        joblib.dump(self.classifier, f"{self.model_dir}/nba_classifier.pkl")
        joblib.dump(self.label_encoder, f"{self.model_dir}/nba_label_encoder.pkl")
        joblib.dump(self.scaler, f"{self.model_dir}/nba_scaler.pkl")
        joblib.dump(self.global_bandit, f"{self.model_dir}/bandit.pkl")
        
        meta = {
            "feature_cols": self.feature_cols,
            "metrics": self.metrics,
            "actions": ACTIONS,
        }
        with open(f"{self.model_dir}/nba_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    def load(self):
        self.classifier = joblib.load(f"{self.model_dir}/nba_classifier.pkl")
        self.label_encoder = joblib.load(f"{self.model_dir}/nba_label_encoder.pkl")
        self.scaler = joblib.load(f"{self.model_dir}/nba_scaler.pkl")
        self.global_bandit = joblib.load(f"{self.model_dir}/bandit.pkl")
        
        with open(f"{self.model_dir}/nba_meta.json") as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        self.metrics = meta["metrics"]
        return self


if __name__ == "__main__":
    df = pd.read_csv("data/master_dataset.csv")
    model = NextBestActionModel(model_dir="models")
    metrics = model.train(df)
    print(json.dumps(metrics, indent=2))
    
    sample = df.sample(1).iloc[0].to_dict()
    rec = model.recommend(sample)
    print("\nSample Recommendation:")
    for r in rec["recommendations"]:
        print(f"  {r['icon']} {r['label']} — confidence: {r['confidence']:.1%}, ROI: {r['expected_roi']:.1f}x")
