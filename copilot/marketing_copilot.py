"""
Module 3: GenAI Marketing Copilot
- RAG pipeline: structured data → context → LLM insights
- Natural language analytics queries
- Auto-generated insights
- Campaign suggestions
"""

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


# ─── Data Context Builder ─────────────────────────────────────────────────────

class DataContextBuilder:
    """
    Builds rich structured context from CSV/dataframe for RAG retrieval.
    No embeddings needed — uses smart data summarization + query routing.
    """
    
    def __init__(self, data_path: str = "data/master_dataset.csv"):
        self.df = None
        self.context_cache = {}
        if os.path.exists(data_path):
            self.load_data(data_path)
    
    def load_data(self, path: str):
        self.df = pd.read_csv(path)
        print(f"✅ Loaded {len(self.df)} customer records")
    
    def get_overview_stats(self) -> str:
        if self.df is None:
            return "No data loaded."
        
        df = self.df
        stats = f"""
=== CUSTOMER BASE OVERVIEW ===
Total Customers: {len(df):,}
Churn Rate: {df['churn_label'].mean():.1%} ({df['churn_label'].sum():,} at risk)
Avg Monthly Spend: ${df['monthly_spend'].mean():.2f}
Total MRR: ${df['monthly_spend'].sum():,.0f}
Avg LTV: ${df['total_lifetime_value'].mean():,.0f}
Avg Tenure: {df['tenure_months'].mean():.1f} months

=== SEGMENT BREAKDOWN ===
{df.groupby('segment').agg(
    Count=('customer_id', 'count'),
    Churn_Rate=('churn_label', 'mean'),
    Avg_Spend=('monthly_spend', 'mean'),
    Avg_LTV=('total_lifetime_value', 'mean')
).round(2).to_string()}

=== CHANNEL DISTRIBUTION ===
{df['channel'].value_counts().to_string()}

=== SPEND TREND ===
{df['spend_trend'].value_counts().to_string()}
"""
        return stats.strip()
    
    def get_churn_analysis(self) -> str:
        if self.df is None:
            return "No data."
        df = self.df
        
        churners = df[df["churn_label"] == 1]
        non_churners = df[df["churn_label"] == 0]
        
        analysis = f"""
=== CHURN ANALYSIS ===
High Risk (>70%): {(df.get('churn_score_true', df['churn_label']) > 0.7).sum():,} customers
Medium Risk (40-70%): {((df.get('churn_score_true', df['churn_label']) > 0.4) & (df.get('churn_score_true', df['churn_label']) <= 0.7)).sum():,} customers
Low Risk (<40%): {(df.get('churn_score_true', df['churn_label']) <= 0.4).sum():,} customers

=== CHURNER PROFILE vs RETAINED ===
Avg Logins (Churners): {churners['logins_last_30d'].mean():.1f}  |  Retained: {non_churners['logins_last_30d'].mean():.1f}
Avg Spend (Churners): ${churners['monthly_spend'].mean():.2f}  |  Retained: ${non_churners['monthly_spend'].mean():.2f}
Avg Feature Usage (Churners): {churners['feature_usage_score'].mean():.1f}  |  Retained: {non_churners['feature_usage_score'].mean():.1f}
Days Inactive (Churners): {churners['days_since_last_login'].mean():.1f}  |  Retained: {non_churners['days_since_last_login'].mean():.1f}
Support Tickets (Churners): {churners['support_tickets_90d'].mean():.1f}  |  Retained: {non_churners['support_tickets_90d'].mean():.1f}

=== CHURN BY SEGMENT ===
{df.groupby('segment')['churn_label'].agg(['mean','sum']).rename(columns={'mean':'Rate','sum':'Count'}).round(3).to_string()}

=== TOP CHURN DRIVERS (Statistical) ===
- Days since last login: {df['days_since_last_login'].corr(df['churn_label']):.3f} correlation with churn
- Low feature usage: {df['feature_usage_score'].corr(df['churn_label']):.3f} correlation
- Support tickets: {df['support_tickets_90d'].corr(df['churn_label']):.3f} correlation  
- Low logins: {df['logins_last_30d'].corr(df['churn_label']):.3f} correlation
"""
        return analysis.strip()
    
    def get_high_value_segments(self) -> str:
        if self.df is None:
            return "No data."
        df = self.df
        
        # RFM-style segmentation
        df_temp = df.copy()
        df_temp["rfm_score"] = (
            (1 / (df_temp["last_purchase_days_ago"] + 1)) * 0.3 +
            (df_temp["purchase_frequency"] / df_temp["purchase_frequency"].max()) * 0.3 +
            (df_temp["monthly_spend"] / df_temp["monthly_spend"].max()) * 0.4
        )
        
        top_segment = df_temp.nlargest(500, "rfm_score")
        
        return f"""
=== HIGH-VALUE CUSTOMER SEGMENT ===
Top 500 customers by RFM Score:
  Avg Monthly Spend: ${top_segment['monthly_spend'].mean():,.2f}
  Avg LTV: ${top_segment['total_lifetime_value'].mean():,.2f}
  Avg Tenure: {top_segment['tenure_months'].mean():.1f} months
  Churn Rate: {top_segment['churn_label'].mean():.1%}
  Most Common Segment: {top_segment['segment'].mode()[0]}
  Most Common Channel: {top_segment['channel'].mode()[0]}
  Email Open Rate: {top_segment['email_open_rate'].mean():.1%}
  Conversion Rate: {top_segment['conversion_rate'].mean():.1%}

Segment Breakdown:
{top_segment['segment'].value_counts().to_string()}

Region Breakdown:
{top_segment['region'].value_counts().to_string()}
"""
    
    def get_campaign_performance(self) -> str:
        if self.df is None:
            return "No data."
        df = self.df
        
        if "total_campaigns_received" not in df.columns:
            return "Campaign data not available."
        
        return f"""
=== CAMPAIGN PERFORMANCE ===
Avg Email Open Rate: {df['email_open_rate'].mean():.1%}
Avg Click-Through Rate: {df['click_through_rate'].mean():.1%}
Avg Conversion Rate: {df['conversion_rate'].mean():.1%}
Total Campaign Revenue: ${df['total_campaign_revenue'].sum():,.0f}
Avg Revenue per Customer from Campaigns: ${df['total_campaign_revenue'].mean():.2f}

Open Rate by Segment:
{df.groupby('segment')['email_open_rate'].mean().round(3).to_string()}

Conversion Rate by Channel:
{df.groupby('channel')['conversion_rate'].mean().round(3).to_string()}

High Converters (>10% conversion):
{(df['conversion_rate'] > 0.10).sum():,} customers ({(df['conversion_rate'] > 0.10).mean():.1%} of base)
"""
    
    def route_query(self, query: str) -> str:
        """Route query to relevant data context"""
        query_lower = query.lower()
        
        contexts = []
        
        # Always include overview for context
        contexts.append(self.get_overview_stats())
        
        if any(w in query_lower for w in ["churn", "retain", "at risk", "leaving", "cancel", "drop"]):
            contexts.append(self.get_churn_analysis())
        
        if any(w in query_lower for w in ["high value", "vip", "best customer", "top customer", "premium", "segment", "ltv", "lifetime"]):
            contexts.append(self.get_high_value_segments())
        
        if any(w in query_lower for w in ["campaign", "email", "conversion", "click", "open rate", "performance", "revenue"]):
            contexts.append(self.get_campaign_performance())
        
        if any(w in query_lower for w in ["spend", "revenue", "mrr", "money", "purchase", "transact"]):
            if self.df is not None:
                df = self.df
                spend_context = f"""
=== SPEND ANALYSIS ===
Revenue at Risk (from churners): ${self.df[self.df['churn_label']==1]['monthly_spend'].sum():,.0f}/month
Monthly Spend Distribution:
  P25: ${df['monthly_spend'].quantile(0.25):.2f}
  Median: ${df['monthly_spend'].median():.2f}
  P75: ${df['monthly_spend'].quantile(0.75):.2f}
  P95: ${df['monthly_spend'].quantile(0.95):.2f}
  
Spend by Region:
{df.groupby('region')['monthly_spend'].mean().round(2).to_string()}
"""
                contexts.append(spend_context)
        
        return "\n\n".join(contexts)
    
    def get_quick_stats(self) -> dict:
        """Return key metrics as dict for dashboard"""
        if self.df is None:
            return {}
        df = self.df
        return {
            "total_customers": len(df),
            "churn_rate": round(float(df["churn_label"].mean()), 4),
            "churners": int(df["churn_label"].sum()),
            "avg_monthly_spend": round(float(df["monthly_spend"].mean()), 2),
            "total_mrr": round(float(df["monthly_spend"].sum()), 2),
            "revenue_at_risk": round(float(df[df["churn_label"]==1]["monthly_spend"].sum()), 2),
            "avg_ltv": round(float(df["total_lifetime_value"].mean()), 2),
            "avg_email_open_rate": round(float(df["email_open_rate"].mean()), 4),
            "avg_conversion_rate": round(float(df["conversion_rate"].mean()), 4),
            "high_risk_count": int((df.get("churn_score_true", df["churn_label"]) > 0.7).sum()),
        }


# ─── Copilot Engine ───────────────────────────────────────────────────────────

class MarketingCopilot:
    """
    GenAI Marketing Copilot using Claude API + RAG data context
    """
    
    SYSTEM_PROMPT = """You are an expert Marketing Analytics AI Copilot for a SaaS company.
You have deep knowledge of:
- Customer churn prediction and retention strategies
- Next-best-action decisioning and campaign optimization  
- Customer segmentation (RFM, behavioral, demographic)
- Marketing analytics: CAC, LTV, cohort analysis, funnel metrics

You are given structured data context retrieved from the company's customer database.
Your job is to analyze this data and provide:
1. Clear, actionable insights
2. Data-driven recommendations
3. Specific numbers and percentages from the data
4. Concrete next steps marketing teams can execute

Be concise but insightful. Format with clear sections. 
Always ground your answers in the provided data context.
When suggesting campaigns, be specific about target segments, channels, and expected outcomes."""
    
    def __init__(self, data_context_builder: DataContextBuilder):
        self.context_builder = data_context_builder
        self.conversation_history = []
    
    async def chat(self, user_message: str, include_suggestions: bool = True) -> dict:
        """Send message to copilot and get response"""
        try:
            import aiohttp
            
            # Build RAG context
            data_context = self.context_builder.route_query(user_message)
            
            # Build prompt
            rag_message = f"""DATA CONTEXT (retrieved from customer database):
{data_context}

USER QUESTION:
{user_message}

{'Please also suggest 2-3 specific campaign actions at the end of your response.' if include_suggestions else ''}"""
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": rag_message
            })
            
            # Call API (will be called from dashboard which has access to API)
            payload = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "system": self.SYSTEM_PROMPT,
                "messages": self.conversation_history[-6:],  # Keep last 6 turns
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    data = await resp.json()
            
            if "content" in data:
                response_text = data["content"][0]["text"]
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                return {
                    "success": True,
                    "response": response_text,
                    "context_used": len(data_context),
                }
            else:
                return {"success": False, "response": str(data), "context_used": 0}
        
        except Exception as e:
            return {"success": False, "response": f"Error: {str(e)}", "context_used": 0}
    
    def sync_chat(self, user_message: str, api_key: str = None) -> dict:
        """Synchronous version for Streamlit"""
        import requests
        
        data_context = self.context_builder.route_query(user_message)
        
        rag_message = f"""DATA CONTEXT (retrieved from customer database):
{data_context}

USER QUESTION:
{user_message}

Please provide data-driven insights and suggest 2-3 specific campaign actions if relevant."""
        
        self.conversation_history.append({
            "role": "user",
            "content": rag_message
        })
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key
        
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "system": self.SYSTEM_PROMPT,
            "messages": self.conversation_history[-6:],
        }
        
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=30,
            )
            data = resp.json()
            
            if "content" in data:
                response_text = data["content"][0]["text"]
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                return {
                    "success": True,
                    "response": response_text,
                    "data_context_length": len(data_context),
                }
            else:
                error_msg = data.get("error", {}).get("message", str(data))
                return {"success": False, "response": f"API Error: {error_msg}"}
        
        except Exception as e:
            return {"success": False, "response": f"Connection error: {str(e)}"}
    
    def generate_auto_insights(self) -> List[str]:
        """Generate automated insights without API call (rule-based)"""
        stats = self.context_builder.get_quick_stats()
        insights = []
        
        if stats.get("churn_rate", 0) > 0.25:
            insights.append(
                f"🚨 **High Churn Alert**: {stats['churn_rate']:.1%} churn rate is above industry benchmark (15-20%). "
                f"${stats['revenue_at_risk']:,.0f}/month in MRR at risk."
            )
        
        insights.append(
            f"💰 **Revenue at Risk**: ${stats.get('revenue_at_risk', 0):,.0f}/month from "
            f"{stats.get('churners', 0):,} at-risk customers. "
            f"Proactive retention could save this revenue."
        )
        
        if stats.get("avg_email_open_rate", 0) < 0.25:
            insights.append(
                f"📧 **Low Email Engagement**: {stats['avg_email_open_rate']:.1%} open rate is below average (30%). "
                f"Consider A/B testing subject lines and send-time optimization."
            )
        elif stats.get("avg_email_open_rate", 0) > 0.35:
            insights.append(
                f"✅ **Strong Email Engagement**: {stats['avg_email_open_rate']:.1%} open rate — leverage this channel more aggressively."
            )
        
        high_risk = stats.get("high_risk_count", 0)
        if high_risk > 0:
            insights.append(
                f"⚠️ **Immediate Action Needed**: {high_risk:,} customers have >70% churn probability. "
                f"Prioritize personal outreach or discount offers for high-LTV customers in this group."
            )
        
        conv = stats.get("avg_conversion_rate", 0)
        if conv < 0.10:
            insights.append(
                f"📊 **Conversion Opportunity**: Average campaign conversion rate is {conv:.1%}. "
                f"Segmenting campaigns by channel preference could lift this by 2-3x."
            )
        
        return insights
    
    def reset_conversation(self):
        self.conversation_history = []


# ─── Preset Query Templates ───────────────────────────────────────────────────

PRESET_QUERIES = [
    {
        "label": "Why is churn high?",
        "query": "Why is our churn rate high? What are the key drivers and which customer segments are most at risk?",
        "icon": "📉",
    },
    {
        "label": "High-value customers",
        "query": "Who are my highest-value customers? What do they have in common and how should I engage them?",
        "icon": "⭐",
    },
    {
        "label": "Campaign strategy",
        "query": "What email and campaign strategy should I prioritize this month based on the data?",
        "icon": "📧",
    },
    {
        "label": "Revenue at risk",
        "query": "How much monthly revenue is at risk from potential churners? Who should I target first?",
        "icon": "💰",
    },
    {
        "label": "Segment to target",
        "query": "Which customer segment should I target for a retention campaign right now and why?",
        "icon": "🎯",
    },
    {
        "label": "Upsell opportunities",
        "query": "Which customers are most likely to respond to an upsell offer? What's the best approach?",
        "icon": "⬆️",
    },
]


if __name__ == "__main__":
    builder = DataContextBuilder("data/master_dataset.csv")
    stats = builder.get_quick_stats()
    print("Quick Stats:", json.dumps(stats, indent=2))
    
    copilot = MarketingCopilot(builder)
    insights = copilot.generate_auto_insights()
    print("\nAuto Insights:")
    for i in insights:
        print(" ", i[:100])
