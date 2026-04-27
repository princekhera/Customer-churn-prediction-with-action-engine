"""
═══════════════════════════════════════════════════════════════════════
  CHURN PREDICTION ENGINE — System 1
  Model: GradientBoostingClassifier (XGBoost-equivalent in sklearn)
  Features: behavioral + transactional
  Output: churn probability + action recommendation + SHAP explanations
═══════════════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import json, os, pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, average_precision_score
)
from sklearn.inspection import permutation_importance

# ── Config ──────────────────────────────────────────────────────────
FEATURES = [
    "login_count_30d", "feature_usage_score", "support_tickets_30d",
    "last_login_days_ago", "session_duration_avg", "total_spend_90d",
    "transactions_90d", "avg_order_value", "spend_trend", "plan_tier",
    "tenure_days", "age",
]
TARGET = "churned"
MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/churn_scaler.pkl"

# ── Action Engine ────────────────────────────────────────────────────
ACTION_THRESHOLDS = {
    "high":   0.70,   # → immediate intervention
    "medium": 0.40,   # → soft nudge
    "low":    0.00,   # → no action needed
}

def recommend_action(churn_prob: float, customer_profile: dict) -> dict:
    """
    Decision logic for next-best retention action.
    Returns action + reason + urgency.
    """
    if churn_prob >= ACTION_THRESHOLDS["high"]:
        # High-value customers → call; others → discount
        if customer_profile.get("plan_tier", 0) >= 2:
            action = "personal_call"
            message = "Schedule a personal call with the customer success team."
        elif customer_profile.get("spend_trend", 0) < -20:
            action = "discount_offer"
            message = "Send a 20% discount valid for 7 days."
        else:
            action = "personalized_email"
            message = "Send a win-back email with feature highlights."
        urgency = "HIGH"

    elif churn_prob >= ACTION_THRESHOLDS["medium"]:
        if customer_profile.get("feature_usage_score", 50) < 30:
            action = "product_tutorial"
            message = "Send onboarding tips and feature walkthrough."
        else:
            action = "personalized_email"
            message = "Send a check-in email with personalized tips."
        urgency = "MEDIUM"

    else:
        action = "no_action"
        message = "Customer is healthy. Continue normal engagement."
        urgency = "LOW"

    return {
        "churn_probability": round(float(churn_prob), 4),
        "churn_risk_level": urgency,
        "recommended_action": action,
        "action_message": message,
    }


# ── SHAP-style Feature Importance Explanation ────────────────────────
def explain_prediction(model, scaler, customer_features: pd.DataFrame) -> list:
    """
    Returns top drivers for a single customer's churn prediction.
    Uses permutation-based feature contribution approximation.
    """
    base_prob = model.predict_proba(scaler.transform(customer_features))[0][1]
    contributions = []

    for feat in FEATURES:
        modified = customer_features.copy()
        modified[feat] = customer_features[feat].mean() if hasattr(customer_features[feat], 'mean') else 0
        modified_prob = model.predict_proba(scaler.transform(modified))[0][1]
        contribution = base_prob - modified_prob
        contributions.append({
            "feature": feat,
            "value": float(customer_features[feat].values[0]),
            "contribution": round(float(contribution), 4),
            "direction": "increases_churn" if contribution > 0 else "decreases_churn"
        })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:6]  # top 6 drivers


# ── Training Pipeline ────────────────────────────────────────────────
class ChurnEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = FEATURES
        self.metrics = {}

    def train(self, data_path: str = "data/churn_data.csv") -> dict:
        print("Loading data...")
        df = pd.read_csv(data_path)

        X = df[FEATURES].fillna(0)
        y = df[TARGET]

        print(f"Dataset: {len(df)} customers | Churn rate: {y.mean():.1%}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        print("Training GradientBoosting (XGBoost-equivalent) model...")
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        self.model.fit(X_train_s, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test_s)
        y_prob = self.model.predict_proba(X_test_s)[:, 1]

        self.metrics = {
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "avg_precision": round(average_precision_score(y_test, y_prob), 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "churn_rate": round(float(y.mean()), 4),
        }

        print("\n── Model Performance ──────────────────────────────────")
        print(f"  ROC-AUC:            {self.metrics['roc_auc']:.4f}")
        print(f"  Avg Precision:      {self.metrics['avg_precision']:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

        # Feature importance
        fi = pd.Series(
            self.model.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)
        print("── Feature Importances ────────────────────────────────")
        for feat, imp in fi.head(8).items():
            bar = "█" * int(imp * 100)
            print(f"  {feat:<28} {bar} {imp:.4f}")

        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"\n✅ Model saved to {MODEL_PATH}")
        return self.metrics

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        return self

    def predict_single(self, customer_dict: dict) -> dict:
        """Full prediction pipeline for a single customer."""
        feat_df = pd.DataFrame([{f: customer_dict.get(f, 0) for f in FEATURES}])
        X_s = self.scaler.transform(feat_df)
        prob = self.model.predict_proba(X_s)[0][1]

        action = recommend_action(prob, customer_dict)
        explanations = explain_prediction(self.model, self.scaler, feat_df)

        return {
            **action,
            "top_churn_drivers": explanations,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict churn for entire customer base."""
        X = df[FEATURES].fillna(0)
        X_s = self.scaler.transform(X)
        probs = self.model.predict_proba(X_s)[:, 1]
        preds = self.model.predict(X_s)

        result = df[["customer_id"] + FEATURES + ["plan", "tenure_days"]].copy() if "customer_id" in df.columns else df[FEATURES].copy()
        result["churn_probability"] = probs.round(4)
        result["churn_predicted"] = preds
        result["risk_level"] = pd.cut(
            probs, bins=[0, 0.4, 0.7, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"]
        )

        # Vectorized action assignment
        def get_action(row):
            return recommend_action(row["churn_probability"], row.to_dict())["recommended_action"]

        result["recommended_action"] = result.apply(get_action, axis=1)
        return result.sort_values("churn_probability", ascending=False)


if __name__ == "__main__":
    engine = ChurnEngine()
    metrics = engine.train()

    print("\n── Sample Prediction ──────────────────────────────────────")
    sample = {
        "login_count_30d": 2,
        "feature_usage_score": 15.0,
        "support_tickets_30d": 3,
        "last_login_days_ago": 45,
        "session_duration_avg": 5.0,
        "total_spend_90d": 80.0,
        "transactions_90d": 2,
        "avg_order_value": 40.0,
        "spend_trend": -35.0,
        "plan_tier": 1,
        "tenure_days": 180,
        "age": 32,
        "plan": "Basic",
    }
    result = engine.predict_single(sample)
    print(json.dumps(result, indent=2))
