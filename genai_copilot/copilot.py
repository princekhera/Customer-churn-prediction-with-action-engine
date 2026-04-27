"""
═══════════════════════════════════════════════════════════════════════
  GENAI MARKETING COPILOT — System 3
  RAG pipeline over structured marketing data
  Backend: CSV/JSON data + LLM (OpenAI or offline rule engine)
  Features: natural language queries, auto-insights, segment analysis
═══════════════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import json, os, re
from datetime import datetime
from typing import Optional

# ── Data Layer (RAG retrieval) ────────────────────────────────────────
class MarketingDataRetriever:
    """
    Loads all structured data and retrieves relevant context
    for a given user query. Acts as the 'R' in RAG.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._weekly = None
        self._segments = None
        self._schema = None

    def load_all(self):
        self._weekly = pd.read_csv(f"{self.data_dir}/weekly_analytics.csv")
        self._segments = pd.read_csv(f"{self.data_dir}/segment_performance.csv")
        with open(f"{self.data_dir}/schema.json") as f:
            self._schema = json.load(f)
        print(f"Loaded: {len(self._weekly)} weeks of analytics, {len(self._segments)} segments")
        return self

    # ── Intent detection ─────────────────────────────────────────────
    def detect_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["drop", "decline", "fall", "down", "decreas", "why"]):
            return "performance_drop"
        elif any(w in q for w in ["high value", "best customer", "top segment", "vip"]):
            return "high_value_segment"
        elif any(w in q for w in ["target", "segment", "who should", "audience"]):
            return "targeting"
        elif any(w in q for w in ["conversion", "convert", "funnel"]):
            return "conversion"
        elif any(w in q for w in ["retention", "churn", "keep"]):
            return "retention"
        elif any(w in q for w in ["revenue", "spend", "money", "roi"]):
            return "revenue"
        elif any(w in q for w in ["channel", "email", "social", "paid"]):
            return "channel"
        elif any(w in q for w in ["trend", "week", "month", "over time"]):
            return "trend"
        else:
            return "general"

    # ── Context retrieval ────────────────────────────────────────────
    def retrieve_context(self, query: str) -> dict:
        intent = self.detect_intent(query)

        ctx = {"intent": intent, "query": query}

        # Always include latest week
        latest = self._weekly.iloc[-1]
        prev = self._weekly.iloc[-2]
        ctx["latest_week"] = latest.to_dict()
        ctx["week_over_week"] = {
            "conversion_change": round((latest["conversions"] - prev["conversions"]) / max(prev["conversions"], 1), 4),
            "revenue_change": round((latest["revenue"] - prev["revenue"]) / max(prev["revenue"], 1), 4),
            "ctr_change": round(latest["ctr"] - prev["ctr"], 4),
        }

        if intent in ("performance_drop", "conversion", "trend"):
            # Last 8 weeks
            recent = self._weekly.tail(8)[["week", "conversions", "revenue", "ctr", "retention_rate"]].to_dict("records")
            ctx["recent_trend"] = recent

            # Detect drops
            drops = []
            for i in range(1, len(self._weekly)):
                row = self._weekly.iloc[i]
                prev_row = self._weekly.iloc[i-1]
                conv_chg = (row["conversions"] - prev_row["conversions"]) / max(prev_row["conversions"], 1)
                if conv_chg < -0.10:
                    drops.append({"week": row["week"], "conversion_drop": round(conv_chg, 3), "revenue": row["revenue"]})
            ctx["notable_drops"] = drops[-3:] if drops else []

        if intent in ("high_value_segment", "targeting", "retention"):
            ctx["segments"] = self._segments.to_dict("records")

        if intent == "channel":
            ctx["channel_distribution"] = self._weekly["top_channel"].value_counts().to_dict()

        if intent == "revenue":
            ctx["revenue_stats"] = {
                "total_52w": round(self._weekly["revenue"].sum(), 2),
                "avg_weekly": round(self._weekly["revenue"].mean(), 2),
                "best_week": self._weekly.loc[self._weekly["revenue"].idxmax(), ["week", "revenue"]].to_dict(),
                "worst_week": self._weekly.loc[self._weekly["revenue"].idxmin(), ["week", "revenue"]].to_dict(),
            }

        return ctx


# ── Rule-based Answer Generator (offline fallback) ────────────────────
class RuleBasedAnswerEngine:
    """
    Generates structured insights without an LLM.
    Used when no API key is available.
    """
    def generate(self, context: dict) -> str:
        intent = context["intent"]
        latest = context["latest_week"]
        wow = context["week_over_week"]

        def pct(v): return f"{v*100:+.1f}%"

        lines = []

        if intent == "performance_drop":
            conv_chg = wow["conversion_change"]
            rev_chg = wow["revenue_change"]
            lines.append(f"📉 **Performance Analysis** (week of {latest['week']})")
            lines.append(f"Conversions are {pct(conv_chg)} vs last week ({latest['conversions']:,} total).")
            lines.append(f"Revenue is {pct(rev_chg)} at ${latest['revenue']:,.0f}.")
            if context.get("notable_drops"):
                drops = context["notable_drops"]
                lines.append(f"\n⚠️ Significant drops detected in {len(drops)} recent weeks.")
                for d in drops:
                    lines.append(f"  • Week {d['week']}: conversions {pct(d['conversion_drop'])}")
            lines.append(f"\n**Possible causes to investigate:**")
            lines.append(f"  1. CTR is {latest['ctr']*100:.2f}% — check ad creative and targeting.")
            lines.append(f"  2. Top channel this week: {latest['top_channel']} — compare vs historical mix.")
            lines.append(f"  3. Retention rate: {latest['retention_rate']*100:.1f}% — any product issues?")

        elif intent in ("high_value_segment", "targeting"):
            segs = context.get("segments", [])
            if segs:
                top = sorted(segs, key=lambda x: x["avg_ltv"], reverse=True)[0]
                at_risk = [s for s in segs if s["churn_rate"] > 0.3]
                lines.append(f"🎯 **Segment Targeting Recommendation**")
                lines.append(f"Highest LTV segment: **{top['segment']}** (avg LTV: ${top['avg_ltv']:,.0f})")
                lines.append(f"  • Email open rate: {top['email_open_rate']*100:.1f}%")
                lines.append(f"  • Monthly spend: ${top['avg_spend_monthly']:,.0f}")
                if at_risk:
                    lines.append(f"\n⚠️ High-churn segments to address: {', '.join(s['segment'] for s in at_risk)}")
                    lines.append(f"Recommended action: targeted win-back campaigns with personalized offers.")

        elif intent == "retention":
            segs = context.get("segments", [])
            if segs:
                high_churn = sorted(segs, key=lambda x: x["churn_rate"], reverse=True)[:2]
                lines.append(f"🔄 **Retention Insights**")
                lines.append(f"Overall retention this week: {latest['retention_rate']*100:.1f}%")
                lines.append(f"\nHighest churn segments:")
                for s in high_churn:
                    lines.append(f"  • {s['segment']}: {s['churn_rate']*100:.1f}% churn rate ({s['customer_count']:,} customers)")
                lines.append(f"\n**Suggested actions:**")
                lines.append(f"  1. Trigger churn prediction model for {high_churn[0]['segment']} customers.")
                lines.append(f"  2. Run personalized email campaign targeting last-active users.")
                lines.append(f"  3. Offer loyalty discount to customers at 60-day inactivity mark.")

        elif intent == "revenue":
            rev = context.get("revenue_stats", {})
            lines.append(f"💰 **Revenue Summary**")
            lines.append(f"  • Total (52 weeks): ${rev.get('total_52w', 0):,.0f}")
            lines.append(f"  • Weekly average: ${rev.get('avg_weekly', 0):,.0f}")
            lines.append(f"  • Best week: {rev.get('best_week', {}).get('week', 'N/A')} — ${rev.get('best_week', {}).get('revenue', 0):,.0f}")
            lines.append(f"  • This week: ${latest['revenue']:,.0f} ({pct(wow['revenue_change'])} WoW)")

        elif intent == "channel":
            dist = context.get("channel_distribution", {})
            lines.append(f"📡 **Channel Performance**")
            lines.append(f"Top-performing channels by frequency (last 52 weeks):")
            for ch, cnt in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  • {ch}: {cnt} weeks as top channel ({cnt/52*100:.0f}%)")
            lines.append(f"\nThis week's top channel: {latest['top_channel']}")

        else:
            lines.append(f"📊 **Marketing Dashboard Summary** — Week of {latest['week']}")
            lines.append(f"  • Impressions: {latest['impressions']:,}")
            lines.append(f"  • Clicks: {latest['clicks']:,} (CTR: {latest['ctr']*100:.2f}%)")
            lines.append(f"  • Conversions: {latest['conversions']:,} ({pct(wow['conversion_change'])} WoW)")
            lines.append(f"  • Revenue: ${latest['revenue']:,.0f} ({pct(wow['revenue_change'])} WoW)")
            lines.append(f"  • Retention: {latest['retention_rate']*100:.1f}%")
            lines.append(f"  • Customer Acq. Cost: ${latest['cac']:.2f}")

        return "\n".join(lines)


# ── OpenAI-powered Generator ──────────────────────────────────────────
class LLMAnswerEngine:
    """
    Uses OpenAI GPT to generate insights from retrieved context.
    Requires OPENAI_API_KEY in environment.
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model
            self.available = True
        except Exception:
            self.available = False

    def generate(self, context: dict) -> str:
        if not self.available:
            return None

        system_prompt = """You are a senior marketing data analyst AI assistant.
You have access to structured marketing analytics data retrieved from the company database.
Provide clear, actionable insights in a professional but conversational tone.
Always include specific numbers from the data. Suggest 2-3 concrete next steps.
Format using markdown with headers and bullet points."""

        user_prompt = f"""User question: {context['query']}

Retrieved data context:
{json.dumps(context, indent=2, default=str)}

Please provide a clear, data-driven answer with specific recommendations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            return None


# ── Auto-Insight Generator ─────────────────────────────────────────────
class AutoInsightEngine:
    """
    Automatically generates weekly insights without prompting.
    """
    def __init__(self, data_dir: str = "data"):
        self.retriever = MarketingDataRetriever(data_dir).load_all()

    def generate_weekly_brief(self) -> str:
        df = self.retriever._weekly
        segs = self.retriever._segments

        latest = df.iloc[-1]
        prev4 = df.tail(5).iloc[:-1]

        insights = []
        insights.append(f"# 📬 Weekly Marketing Intelligence Brief")
        insights.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        # Conversion trend
        conv_avg = prev4["conversions"].mean()
        conv_delta = (latest["conversions"] - conv_avg) / conv_avg
        emoji = "🟢" if conv_delta > 0 else "🔴"
        insights.append(f"## {emoji} Conversion Performance")
        insights.append(f"This week: **{latest['conversions']:,} conversions** ({conv_delta*100:+.1f}% vs 4-week avg)")

        # Revenue insight
        rev_avg = prev4["revenue"].mean()
        rev_delta = (latest["revenue"] - rev_avg) / rev_avg
        insights.append(f"\n## 💰 Revenue")
        insights.append(f"${latest['revenue']:,.0f} this week ({rev_delta*100:+.1f}% vs avg)")

        # At-risk segment alert
        if segs is not None:
            at_risk = segs[segs["churn_rate"] > 0.30]
            if not at_risk.empty:
                insights.append(f"\n## ⚠️ Churn Alert")
                for _, row in at_risk.iterrows():
                    insights.append(f"  **{row['segment']}**: {row['churn_rate']*100:.0f}% churn rate — {row['customer_count']:,} customers at risk")

        # Recommendation
        insights.append(f"\n## 🎯 Top Recommendations")
        if conv_delta < -0.05:
            insights.append("1. Investigate top-of-funnel — CTR dropped, review creative and audience targeting.")
        else:
            insights.append("1. Scale spend on best-performing channel to capitalize on conversion momentum.")
        insights.append("2. Run next-best-action model on At Risk segment this week.")
        insights.append("3. A/B test subject lines for the re-engagement email campaign.")

        return "\n".join(insights)


# ── Main Copilot Interface ─────────────────────────────────────────────
class MarketingCopilot:
    def __init__(self, data_dir: str = "data"):
        self.retriever = MarketingDataRetriever(data_dir).load_all()
        self.llm = LLMAnswerEngine()
        self.rule_engine = RuleBasedAnswerEngine()
        self.history = []

    def ask(self, question: str) -> str:
        context = self.retriever.retrieve_context(question)

        # Try LLM first; fall back to rule engine
        answer = None
        if self.llm.available:
            answer = self.llm.generate(context)

        if not answer:
            answer = self.rule_engine.generate(context)

        self.history.append({"question": question, "answer": answer, "intent": context["intent"]})
        return answer

    def auto_brief(self) -> str:
        engine = AutoInsightEngine()
        return engine.generate_weekly_brief()


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    copilot = MarketingCopilot()

    print("=" * 65)
    print("GENAI MARKETING COPILOT — Demo")
    print("=" * 65)

    questions = [
        "Why did conversion drop last week?",
        "Who are my high-value users?",
        "What segment should I target?",
        "How is revenue trending?",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        print("-" * 55)
        answer = copilot.ask(q)
        print(answer)

    print("\n" + "=" * 65)
    print("AUTO-GENERATED WEEKLY BRIEF")
    print("=" * 65)
    print(copilot.auto_brief())
