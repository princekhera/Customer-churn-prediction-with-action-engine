"""
═══════════════════════════════════════════════════════════════════════
  NEXT-BEST-ACTION ENGINE — System 2
  Model: Multi-class GradientBoosting + Contextual Bandit simulation
  Output: optimal action per customer + expected conversion + confidence
═══════════════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import json, os, pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier

ACTIONS = ["send_email", "push_notification", "upsell_offer", "do_nothing"]

FEATURES = [
    "login_count_30d", "feature_usage_score", "last_login_days_ago",
    "total_spend_90d", "transactions_90d", "avg_order_value",
    "spend_trend", "plan_tier", "tenure_days", "age",
    "email_opens_30d", "push_clicks_30d", "promo_redeemed_90d",
    "calls_received_90d", "last_campaign_days_ago", "last_campaign_converted",
]

MODEL_PATH = "models/nba_model.pkl"
SCALER_PATH = "models/nba_scaler.pkl"
ENCODER_PATH = "models/nba_encoder.pkl"

# ── Contextual Bandit Reward Simulation ─────────────────────────────
class ContextualBandit:
    """
    Epsilon-greedy contextual bandit for online action optimization.
    Simulates learning from customer response feedback.
    """
    def __init__(self, actions: list, epsilon: float = 0.15):
        self.actions = actions
        self.epsilon = epsilon
        self.counts = {a: 0 for a in actions}
        self.rewards = {a: [] for a in actions}
        self.q_values = {a: 0.5 for a in actions}  # optimistic init

    def select_action(self, context: dict = None) -> str:
        """Epsilon-greedy selection."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)  # explore
        return max(self.q_values, key=self.q_values.get)  # exploit

    def update(self, action: str, reward: float):
        """Update Q-value with incremental mean."""
        self.counts[action] += 1
        self.rewards[action].append(reward)
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n

    def simulate(self, n_rounds: int = 1000) -> pd.DataFrame:
        """Simulate n_rounds of bandit feedback to show convergence."""
        true_rewards = {
            "send_email": 0.25,
            "push_notification": 0.18,
            "upsell_offer": 0.35,
            "do_nothing": 0.05,
        }
        history = []
        cumulative_reward = 0

        for t in range(n_rounds):
            action = self.select_action()
            reward = np.random.binomial(1, true_rewards[action])
            self.update(action, float(reward))
            cumulative_reward += reward

            if t % 50 == 0:
                history.append({
                    "round": t,
                    "action_selected": action,
                    "cumulative_reward": cumulative_reward,
                    "q_values": dict(self.q_values),
                })

        return pd.DataFrame(history)

    def get_summary(self) -> dict:
        return {
            "action_counts": self.counts,
            "q_values": {k: round(v, 4) for k, v in self.q_values.items()},
            "estimated_best_action": max(self.q_values, key=self.q_values.get),
        }


# ── NBA Model ────────────────────────────────────────────────────────
class NBAEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.bandit = ContextualBandit(ACTIONS)
        self.metrics = {}

    def train(self, data_path: str = "data/nba_data.csv") -> dict:
        print("Loading NBA dataset...")
        df = pd.read_csv(data_path)

        X = df[FEATURES].fillna(0)
        y = self.label_encoder.fit_transform(df["optimal_action"])

        print(f"Dataset: {len(df)} customers | Actions: {self.label_encoder.classes_.tolist()}")
        print(f"Action distribution:\n{df['optimal_action'].value_counts().to_string()}\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler.fit(X_train)
        Xtr = self.scaler.transform(X_train)
        Xte = self.scaler.transform(X_test)

        print("Training multi-class GradientBoosting NBA model...")
        self.model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=15, random_state=42
        )
        self.model.fit(Xtr, y_train)

        y_pred = self.model.predict(Xte)
        action_names = self.label_encoder.classes_

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        }

        print("── Model Performance ──────────────────────────────────")
        print(f"  Accuracy:     {self.metrics['accuracy']:.4f}")
        print(f"  Macro F1:     {self.metrics['macro_f1']:.4f}")
        print(classification_report(y_test, y_pred, target_names=action_names))

        # Run bandit simulation
        print("Running contextual bandit simulation (1000 rounds)...")
        bandit_history = self.bandit.simulate(1000)
        bandit_summary = self.bandit.get_summary()
        print(f"  Bandit best action learned: {bandit_summary['estimated_best_action']}")
        print(f"  Q-values: {bandit_summary['q_values']}")

        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb") as f: pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)
        with open(ENCODER_PATH, "wb") as f: pickle.dump(self.label_encoder, f)

        print(f"\n✅ NBA model saved.")
        return {**self.metrics, "bandit_summary": bandit_summary}

    def load(self):
        with open(MODEL_PATH, "rb") as f: self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f: self.label_encoder = pickle.load(f)
        return self

    def predict_single(self, customer_dict: dict) -> dict:
        feat_df = pd.DataFrame([{f: customer_dict.get(f, 0) for f in FEATURES}])
        X_s = self.scaler.transform(feat_df)

        probs = self.model.predict_proba(X_s)[0]
        pred_idx = np.argmax(probs)
        pred_action = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

        all_actions = self.label_encoder.inverse_transform(range(len(probs)))
        action_scores = {str(a): round(float(p), 4) for a, p in zip(all_actions, probs)}

        # Message templates
        messages = {
            "send_email": "Send a personalized re-engagement email.",
            "push_notification": "Trigger a push notification with a relevant offer.",
            "upsell_offer": "Present an upsell offer for a higher tier plan.",
            "do_nothing": "No action needed — customer is engaged.",
        }

        return {
            "recommended_action": pred_action,
            "confidence": round(confidence, 4),
            "action_message": messages.get(pred_action, ""),
            "all_action_scores": action_scores,
            "bandit_recommendation": self.bandit.select_action(customer_dict),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].fillna(0)
        X_s = self.scaler.transform(X)

        probs = self.model.predict_proba(X_s)
        pred_indices = np.argmax(probs, axis=1)
        pred_actions = self.label_encoder.inverse_transform(pred_indices)
        confidence = probs.max(axis=1)

        result = df[["customer_id"]].copy() if "customer_id" in df.columns else pd.DataFrame()
        result["recommended_action"] = pred_actions
        result["confidence"] = confidence.round(4)
        return result


if __name__ == "__main__":
    engine = NBAEngine()
    metrics = engine.train()

    print("\n── Sample NBA Prediction ───────────────────────────────────")
    sample = {
        "login_count_30d": 18,
        "feature_usage_score": 72.0,
        "last_login_days_ago": 3,
        "total_spend_90d": 450.0,
        "transactions_90d": 8,
        "avg_order_value": 56.25,
        "spend_trend": 45.0,
        "plan_tier": 1,
        "tenure_days": 365,
        "age": 35,
        "email_opens_30d": 4,
        "push_clicks_30d": 1,
        "promo_redeemed_90d": 2,
        "calls_received_90d": 0,
        "last_campaign_days_ago": 14,
        "last_campaign_converted": 1,
    }
    result = engine.predict_single(sample)
    print(json.dumps(result, indent=2))
