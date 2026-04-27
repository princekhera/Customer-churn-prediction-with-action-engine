"""Streamlit Dashboard for Marketing AI Platform"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Marketing AI Platform", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0e1117; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_churn():
    from churn_engine.churn_model import ChurnEngine
    try: return ChurnEngine().load()
    except: return None

@st.cache_resource
def load_nba():
    from nba_engine.nba_model import NBAEngine
    try: return NBAEngine().load()
    except: return None

@st.cache_resource
def load_copilot():
    from genai_copilot.copilot import MarketingCopilot
    try: return MarketingCopilot()
    except: return None

@st.cache_data
def load_data():
    return (pd.read_csv("data/churn_data.csv"), pd.read_csv("data/nba_data.csv"),
            pd.read_csv("data/weekly_analytics.csv"), pd.read_csv("data/segment_performance.csv"))

st.title("🎯 Marketing AI Platform")
st.caption("Churn Prediction · Next-Best-Action · GenAI Copilot")

with st.sidebar:
    page = st.radio("Navigate", ["📊 Overview", "🔮 Churn Engine", "⚡ Next-Best-Action", "🤖 GenAI Copilot"])

churn_engine = load_churn()
nba_engine = load_nba()
copilot = load_copilot()

try:
    churn_df, nba_df, weekly_df, segment_df = load_data()
except Exception as e:
    st.error(f"Run `python data/generate_data.py` first. Error: {e}"); st.stop()

# ── OVERVIEW ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    latest = weekly_df.iloc[-1]; prev = weekly_df.iloc[-2]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Conversions", f"{latest['conversions']:,}", f"{latest['conversions']-prev['conversions']:+,}")
    c2.metric("Revenue", f"${latest['revenue']:,.0f}", f"${latest['revenue']-prev['revenue']:+,.0f}")
    c3.metric("CTR", f"{latest['ctr']*100:.2f}%")
    c4.metric("Retention", f"{latest['retention_rate']*100:.1f}%")
    c5.metric("Overall Churn Rate", f"{churn_df['churned'].mean()*100:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=weekly_df["week"], y=weekly_df["conversions"], name="Conversions", line=dict(color="#4fc3f7")), secondary_y=False)
        fig.add_trace(go.Scatter(x=weekly_df["week"], y=weekly_df["revenue"], name="Revenue", line=dict(color="#81c784", dash="dot")), secondary_y=True)
        fig.update_layout(template="plotly_dark", height=300, title="Weekly Conversions & Revenue")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(segment_df, x="segment", y="churn_rate", color="churn_rate",
                      color_continuous_scale="RdYlGn_r", template="plotly_dark", height=300, title="Churn Rate by Segment")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        ac = nba_df["optimal_action"].value_counts()
        fig3 = px.pie(values=ac.values, names=ac.index, template="plotly_dark", height=280, title="NBA Action Distribution",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        ch = weekly_df["top_channel"].value_counts().reset_index()
        ch.columns = ["channel","count"]
        fig4 = px.bar(ch, x="channel", y="count", color="channel", template="plotly_dark", height=280, title="Top Channel Frequency")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Segment Performance")
    st.dataframe(segment_df.style.format({"avg_ltv":"${:,.0f}","churn_rate":"{:.1%}","avg_spend_monthly":"${:.0f}","email_open_rate":"{:.1%}","push_ctr":"{:.2%}"}), use_container_width=True)

# ── CHURN ENGINE ──────────────────────────────────────────────────────
elif page == "🔮 Churn Engine":
    st.markdown("## 🔮 Churn Prediction Engine")
    if not churn_engine:
        st.error("Run: `python churn_engine/churn_model.py`"); st.stop()

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Analysis"])
    with tab1:
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Behavioral**")
            login_count = st.slider("Logins 30d", 0, 30, 5)
            feature_usage = st.slider("Feature Usage", 0.0, 100.0, 20.0)
            last_login = st.slider("Last Login (days ago)", 0, 120, 45)
            support_tickets = st.number_input("Support Tickets", 0, 20, 2)
            session_dur = st.number_input("Avg Session (min)", 0.0, 120.0, 8.0)
        with c2:
            st.markdown("**Transactional**")
            total_spend = st.number_input("Spend 90d ($)", 0.0, 5000.0, 80.0)
            transactions = st.number_input("Transactions 90d", 0, 50, 2)
            avg_order = st.number_input("Avg Order ($)", 0.0, 1000.0, 40.0)
            spend_trend = st.number_input("Spend Trend (+/-)", -500.0, 500.0, -30.0)
        with c3:
            st.markdown("**Account**")
            plan = st.selectbox("Plan", ["Free","Basic","Pro","Enterprise"])
            plan_map = {"Free":0,"Basic":1,"Pro":2,"Enterprise":3}
            tenure = st.number_input("Tenure (days)", 0, 2000, 180)
            age = st.number_input("Age", 18, 80, 32)

        if st.button("🔮 Predict", type="primary", use_container_width=True):
            cust = {"login_count_30d":login_count,"feature_usage_score":feature_usage,"support_tickets_30d":support_tickets,
                    "last_login_days_ago":last_login,"session_duration_avg":session_dur,"total_spend_90d":total_spend,
                    "transactions_90d":transactions,"avg_order_value":avg_order,"spend_trend":spend_trend,
                    "plan_tier":plan_map[plan],"plan":plan,"tenure_days":tenure,"age":age}
            res = churn_engine.predict_single(cust)
            prob = res["churn_probability"]
            cc1,cc2,cc3 = st.columns(3)
            cc1.metric("Churn Probability", f"{prob*100:.1f}%")
            cc2.metric("Risk Level", res["churn_risk_level"])
            cc3.metric("Action", res["recommended_action"].replace("_"," ").title())

            gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={"text":"Churn Risk","font":{"color":"white"}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#ff4b6e" if prob>0.7 else "#ffb347" if prob>0.4 else "#4caf50"},
                       "steps":[{"range":[0,40],"color":"#1a3a2a"},{"range":[40,70],"color":"#3a2d0d"},{"range":[70,100],"color":"#3a0d15"}]}))
            gauge.update_layout(template="plotly_dark", height=260, margin=dict(t=40,b=20))
            st.plotly_chart(gauge, use_container_width=True)

            drivers = res.get("top_churn_drivers",[])
            if drivers:
                ddf = pd.DataFrame(drivers)
                colors = ["#ff4b6e" if d>0 else "#4caf50" for d in ddf["contribution"]]
                fig_s = go.Figure(go.Bar(x=ddf["contribution"], y=ddf["feature"], orientation="h", marker_color=colors))
                fig_s.update_layout(template="plotly_dark", height=300, title="Feature Contributions (SHAP-style)", margin=dict(l=160,t=50))
                st.plotly_chart(fig_s, use_container_width=True)
            st.info(f"📬 {res['action_message']}")

    with tab2:
        n = st.slider("Sample size", 100, 2000, 500)
        if st.button("Run Batch Scoring", type="primary"):
            with st.spinner("Scoring..."):
                br = churn_engine.predict_batch(churn_df.sample(n, random_state=42))
            c1,c2,c3 = st.columns(3)
            c1.metric("🔴 High Risk", int((br["risk_level"]=="HIGH").sum()))
            c2.metric("🟡 Medium Risk", int((br["risk_level"]=="MEDIUM").sum()))
            c3.metric("🟢 Low Risk", int((br["risk_level"]=="LOW").sum()))
            fig = px.histogram(br, x="churn_probability", color="risk_level", nbins=40,
                               color_discrete_map={"HIGH":"#ff4b6e","MEDIUM":"#ffb347","LOW":"#4caf50"},
                               template="plotly_dark", title="Churn Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(br.nlargest(20,"churn_probability")[["customer_id","churn_probability","risk_level","recommended_action"]]
                         .style.format({"churn_probability":"{:.1%}"}), use_container_width=True)

# ── NBA ENGINE ────────────────────────────────────────────────────────
elif page == "⚡ Next-Best-Action":
    st.markdown("## ⚡ Next-Best-Action Engine")
    if not nba_engine:
        st.error("Run: `python nba_engine/nba_model.py`"); st.stop()

    tab1, tab2 = st.tabs(["Single Recommendation", "Segment Analysis"])
    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Profile**")
            lc = st.slider("Logins 30d", 0, 30, 15, key="nl")
            fu = st.slider("Feature Usage", 0.0, 100.0, 65.0, key="nf")
            ll = st.slider("Last Login (days ago)", 0, 120, 5, key="nll")
            sp = st.number_input("Spend 90d ($)", 0.0, 5000.0, 420.0, key="ns")
            st = st.number_input("Spend Trend", -500.0, 500.0, 40.0, key="nst")
        with c2:
            st2 = st
            st2.markdown("**Engagement**")
            eo = st2.number_input("Email Opens 30d", 0, 30, 4)
            pc = st2.number_input("Push Clicks 30d", 0, 30, 1)
            pr = st2.number_input("Promos Redeemed", 0, 20, 2)
            lct = st2.selectbox("Last Campaign", ["none","email","push","upsell_offer","discount"])

        if st2.button("⚡ Recommend", type="primary", use_container_width=True):
            cust = {"login_count_30d":lc,"feature_usage_score":fu,"last_login_days_ago":ll,"total_spend_90d":sp,
                    "transactions_90d":6,"avg_order_value":70.0,"spend_trend":sp,"plan_tier":1,"tenure_days":300,
                    "age":35,"email_opens_30d":eo,"push_clicks_30d":pc,"promo_redeemed_90d":pr,"calls_received_90d":0,
                    "last_campaign_type":lct,"last_campaign_days_ago":14,"last_campaign_converted":1,"support_tickets_30d":0}
            res = nba_engine.predict_single(cust)
            action_colors={"send_email":"#4fc3f7","push_notification":"#ce93d8","upsell_offer":"#81c784","do_nothing":"#9e9e9e"}
            a = res["recommended_action"]
            c = action_colors.get(a,"#fff")
            st2.markdown(f"### Action: <span style='color:{c}'>{a.replace('_',' ').upper()}</span>", unsafe_allow_html=True)
            st2.metric("Confidence", f"{res['confidence']*100:.1f}%")
            st2.info(res["action_message"])
            scores = res["all_action_scores"]
            fig = go.Figure(go.Bar(x=list(scores.values()),y=list(scores.keys()),orientation="h",
                                   marker_color=[action_colors.get(k,"#aaa") for k in scores]))
            fig.update_layout(template="plotly_dark",height=260,title="Action Scores",margin=dict(l=160,t=40))
            st2.plotly_chart(fig, use_container_width=True)

    with tab2:
        n2 = st2.slider("Sample size", 200, 2000, 600, key="nb2")
        if st2.button("Analyze", type="primary"):
            res = nba_engine.predict_batch(nba_df.sample(n2, random_state=42))
            c1,c2 = st2.columns(2)
            with c1:
                ad = res["recommended_action"].value_counts().reset_index(); ad.columns=["action","count"]
                st2.plotly_chart(px.pie(ad, values="count", names="action", template="plotly_dark",
                                       title="Action Distribution", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            with c2:
                cb = res.groupby("recommended_action")["confidence"].mean().reset_index()
                st2.plotly_chart(px.bar(cb, x="recommended_action", y="confidence", color="recommended_action",
                                       template="plotly_dark", title="Avg Confidence by Action"), use_container_width=True)

# ── COPILOT ────────────────────────────────────────────────────────────
elif page == "🤖 GenAI Copilot":
    st.markdown("## 🤖 GenAI Marketing Copilot")
    if not copilot:
        st.error("Data not found."); st.stop()

    ec = st.columns(4)
    examples = ["Why did conversion drop last week?","Who are my high-value users?","What segment should I target?","How is revenue trending?"]
    for i,(col,q) in enumerate(zip(ec,examples)):
        if col.button(q, key=f"eq{i}"): st.session_state["cq"]=q

    if "chat_history" not in st.session_state: st.session_state["chat_history"]=[]

    query = st.text_input("Ask:", value=st.session_state.get("cq",""), placeholder="e.g. Why did conversion drop?")
    c1,c2 = st.columns([3,1])
    submit = c1.button("🔍 Ask", type="primary", use_container_width=True)
    brief_btn = c2.button("📬 Weekly Brief", use_container_width=True)

    if submit and query:
        with st.spinner("Analyzing..."):
            answer = copilot.ask(query)
            intent = copilot.history[-1]["intent"] if copilot.history else "general"
        st.session_state["chat_history"].append({"q":query,"a":answer,"intent":intent})
        if "cq" in st.session_state: del st.session_state["cq"]

    if brief_btn:
        with st.spinner("Generating brief..."):
            brief_text = copilot.auto_brief()
        st.session_state["chat_history"].append({"q":"📬 Weekly Brief","a":brief_text,"intent":"general"})

    for item in reversed(st.session_state["chat_history"]):
        st.markdown(f"**❓ {item['q']}** — *Intent: `{item['intent']}`*")
        st.markdown(item["a"])
        st.divider()

    with st.expander("📊 Data Explorer"):
        metric = st.selectbox("Metric", ["conversions","revenue","ctr","retention_rate"])
        fig = px.line(weekly_df, x="week", y=metric, template="plotly_dark", markers=True)
        fig.update_traces(line=dict(color="#4fc3f7", width=2))
        st.plotly_chart(fig, use_container_width=True)
