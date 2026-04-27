"""FastAPI backend — Churn + NBA + GenAI Copilot"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd

from churn_engine.churn_model import ChurnEngine
from nba_engine.nba_model import NBAEngine
from genai_copilot.copilot import MarketingCopilot

app = FastAPI(title="Marketing AI Platform", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading models...")
try: churn_engine = ChurnEngine().load(); print("Churn model OK")
except: churn_engine = None; print("Churn model not found")
try: nba_engine = NBAEngine().load(); print("NBA model OK")
except: nba_engine = None; print("NBA model not found")
try: copilot = MarketingCopilot(); print("Copilot OK")
except: copilot = None; print("Copilot not found")

class CustomerFeatures(BaseModel):
    customer_id: Optional[str]="unknown"
    login_count_30d: float=10; feature_usage_score: float=50
    support_tickets_30d: float=0; last_login_days_ago: float=7
    session_duration_avg: float=15; total_spend_90d: float=150
    transactions_90d: float=5; avg_order_value: float=30
    spend_trend: float=0; plan_tier: int=1; plan: str="Basic"
    tenure_days: int=180; age: int=35
    email_opens_30d: float=2; push_clicks_30d: float=1
    promo_redeemed_90d: float=0; calls_received_90d: float=0
    last_campaign_type: str="none"; last_campaign_days_ago: float=30
    last_campaign_converted: int=0

class CopilotQuery(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"churn": bool(churn_engine), "nba": bool(nba_engine), "copilot": bool(copilot)}

@app.post("/churn/predict")
def predict_churn(c: CustomerFeatures):
    if not churn_engine: raise HTTPException(503, "Model not loaded")
    r = churn_engine.predict_single(c.model_dump()); r["customer_id"]=c.customer_id; return r

@app.post("/nba/recommend")
def recommend(c: CustomerFeatures):
    if not nba_engine: raise HTTPException(503, "Model not loaded")
    r = nba_engine.predict_single(c.model_dump()); r["customer_id"]=c.customer_id; return r

@app.post("/copilot/ask")
def ask(q: CopilotQuery):
    if not copilot: raise HTTPException(503, "Copilot not loaded")
    answer = copilot.ask(q.question)
    return {"question":q.question,"answer":answer}

@app.get("/copilot/brief")
def brief():
    if not copilot: raise HTTPException(503)
    return {"brief": copilot.auto_brief()}

@app.post("/full-profile")
def full_profile(c: CustomerFeatures):
    result = {"customer_id": c.customer_id}
    if churn_engine: result["churn"] = churn_engine.predict_single(c.model_dump())
    if nba_engine: result["nba"] = nba_engine.predict_single(c.model_dump())
    return result

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
