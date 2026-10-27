"""
Employee Attrition & Retention Prediction System.
FastAPI Backend Application & Interactive Mobile Web Interface.
Loads trained Scikit-Learn Random Forest pipeline (model.pkl) and serves real-time inference.
"""

import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, model_validator, field_validator
from sklearn.base import BaseEstimator, TransformerMixin
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# =====================================================================
# 1. DOMAIN FEATURE ENGINEERING TRANSFORMER
# =====================================================================

class EmployeeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain feature engineering for workplace attrition modeling:
    - tenure_ratio: years_at_company / max(total_working_years, 1)
    - promotion_stagnation_ratio: years_since_last_promotion / max(years_at_company, 1)
    - income_per_exp_year: monthly_income / max(total_working_years, 1)
    - wellbeing_index: (job_satisfaction * 0.55) + (work_life_balance * 0.45)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            raise ValueError("Input to EmployeeFeatureEngineer must be a pandas DataFrame")

        tot_exp = np.maximum(X_copy["total_working_years"].values, 1)
        tenure = np.maximum(X_copy["years_at_company"].values, 1)
        prom_years = X_copy["years_since_last_promotion"].values
        income = X_copy["monthly_income"].values
        sat = X_copy["job_satisfaction"].values
        wlb = X_copy["work_life_balance"].values

        X_copy["tenure_ratio"] = np.round(tenure / tot_exp, 3)
        X_copy["promotion_stagnation_ratio"] = np.round(prom_years / tenure, 3)
        X_copy["income_per_exp_year"] = np.round(income / tot_exp, 2)
        X_copy["wellbeing_index"] = np.round((sat * 0.55) + (wlb * 0.45), 2)

        return X_copy


# =====================================================================
# 2. PYDANTIC SCHEMAS & INPUT VALIDATION
# =====================================================================

VALID_DEPTS = [
    "Technology", "Sales & Marketing", "Finance & Operations",
    "Human Resources", "Healthcare & R&D"
]

class EmployeePredictionRequest(BaseModel):
    age: int = Field(..., gt=17, le=80, description="Employee age in years")
    department: str = Field(..., description="Department name")
    job_role: str = Field(..., description="Job role title")
    monthly_income: float = Field(..., gt=500, description="Monthly base salary in USD")
    total_working_years: int = Field(..., ge=0, description="Total professional working experience")
    years_at_company: int = Field(..., ge=0, description="Tenure at current enterprise")
    years_since_last_promotion: int = Field(..., ge=0, description="Years elapsed since last promotion")
    overtime: str = Field(..., description="Regular expectation of overtime (Yes / No)")
    distance_from_home_km: float = Field(..., ge=0, description="Commute distance in kilometers")
    job_satisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction rating (1 to 4)")
    work_life_balance: int = Field(..., ge=1, le=4, description="Work-life balance rating (1 to 4)")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "overtime" in values and isinstance(values["overtime"], str):
                values["overtime"] = values["overtime"].strip().capitalize()
            if "department" in values and isinstance(values["department"], str):
                values["department"] = values["department"].strip()
            if "job_role" in values and isinstance(values["job_role"], str):
                values["job_role"] = values["job_role"].strip()
        return values

    @field_validator("overtime")
    @classmethod
    def validate_overtime(cls, v: str) -> str:
        if v not in ["Yes", "No"]:
            raise ValueError("overtime must be 'Yes' or 'No'")
        return v


class EmployeePredictionResponse(BaseModel):
    predicted_risk_level: str
    risk_score_percentage: float
    confidence: float
    probabilities: Dict[str, float]
    status_tag: str
    key_drivers: List[str]
    retention_recommendations: List[str]
    summary: Dict[str, Any]


# =====================================================================
# 3. PREDICTOR SINGLETON
# =====================================================================

class AttritionPredictor:
    def __init__(self, model_file: str = "model.pkl"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, model_file)
        self.pipeline = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                print(f"Loaded ML pipeline from: {self.model_path}")
            except Exception:
                with open(self.model_path, "rb") as f:
                    self.pipeline = pickle.load(f)
                print(f"Loaded ML pipeline via pickle from: {self.model_path}")
        else:
            print(f"Warning: {self.model_path} not found.")

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def predict(self, req: EmployeePredictionRequest) -> EmployeePredictionResponse:
        if not self.is_loaded:
            self.load_model()
            if not self.is_loaded:
                raise RuntimeError("Model pipeline failed to load.")

        input_df = pd.DataFrame([{
            "age": req.age,
            "department": req.department,
            "job_role": req.job_role,
            "monthly_income": float(req.monthly_income),
            "total_working_years": int(req.total_working_years),
            "years_at_company": int(req.years_at_company),
            "years_since_last_promotion": int(req.years_since_last_promotion),
            "overtime": req.overtime,
            "distance_from_home_km": float(req.distance_from_home_km),
            "job_satisfaction": int(req.job_satisfaction),
            "work_life_balance": int(req.work_life_balance)
        }])

        pred_class = str(self.pipeline.predict(input_df)[0])

        probabilities = {}
        confidence = 0.85

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(input_df)[0]
            classes = list(self.pipeline.classes_)
            for c, p in zip(classes, probs):
                probabilities[str(c)] = round(float(p), 4)
            confidence = probabilities.get(pred_class, 0.85)

        # Calculate estimated flight risk percentage
        # Weighted probability: Low Risk (weight 0.15), Moderate Risk (weight 0.50), High Risk (weight 0.90)
        p_low = probabilities.get("Low Risk", 0.33)
        p_mod = probabilities.get("Moderate Risk", 0.33)
        p_high = probabilities.get("High Risk", 0.33)
        risk_pct = round((p_low * 15.0) + (p_mod * 50.0) + (p_high * 90.0), 1)

        # Synthesize key influencing drivers
        drivers = []
        if req.overtime == "Yes":
            drivers.append("Frequent overtime schedules increasing burnout risk")
        if req.job_satisfaction <= 2:
            drivers.append("Low day-to-day role sentiment and morale")
        if req.work_life_balance <= 2:
            drivers.append("Imbalanced work-life equilibrium")
        if req.years_since_last_promotion >= 3:
            drivers.append(f"{req.years_since_last_promotion} years without promotion advancement")
        if req.distance_from_home_km >= 25:
            drivers.append(f"Long commute distance ({req.distance_from_home_km} km)")
        if not drivers:
            drivers.append("Positive role alignment and competitive compensation")
            drivers.append("Healthy personal-professional work-life equilibrium")

        # Contextual HR Recommendations
        recs = []
        if "High" in pred_class:
            tag = "Flight Risk 🚨"
            recs = [
                "Schedule an immediate 1-on-1 career aspirations check-in.",
                "Review market salary benchmarking and merit-based adjustment.",
                "Enforce strict work-hour boundaries to relieve overtime fatigue.",
                "Explore horizontal or vertical internal mobility opportunities."
            ]
        elif "Moderate" in pred_class:
            tag = "Watch List ⚠️"
            recs = [
                "Conduct quarterly career progression & promotion velocity review.",
                "Offer hybrid or flexible commuting arrangements if distance is high.",
                "Pair with a senior executive mentor for professional development.",
                "Assess project diversity to re-energize engagement."
            ]
        else: # Low Risk
            tag = "Thriving Talent 🌱"
            recs = [
                "Maintain ongoing recognition and high-performance incentives.",
                "Engage in leadership pipeline and technical mentorship roles.",
                "Keep tracking annual satisfaction and compensation growth."
            ]

        return EmployeePredictionResponse(
            predicted_risk_level=pred_class,
            risk_score_percentage=risk_pct,
            confidence=round(float(confidence), 3),
            probabilities=probabilities,
            status_tag=tag,
            key_drivers=drivers,
            retention_recommendations=recs,
            summary={
                "age": str(req.age),
                "department": req.department,
                "job_role": req.job_role,
                "monthly_income": f"${req.monthly_income:,.0f}",
                "tenure": f"{req.years_at_company} yrs",
                "overtime": req.overtime,
                "satisfaction": f"{req.job_satisfaction} / 4"
            }
        )

predictor = AttritionPredictor()

# =====================================================================
# 4. FASTAPI WEB SERVICE & REST ENDPOINTS
# =====================================================================

app = FastAPI(
    title="Employee Attrition & Retention Prediction API",
    description="Machine Learning REST service predicting employee turnover risk categories.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", summary="Interactive Web UI")
def root(request: Request):
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header and "text/html" not in accept_header:
        return JSONResponse(content={
            "status": "online",
            "service": "Employee Attrition & Retention Prediction Engine",
            "endpoints": {
                "ui": "/",
                "health": "/health",
                "model_info": "/model-info",
                "predict": "/predict",
                "docs": "/docs"
            }
        })
    return HTMLResponse(content=EMBEDDED_UI_HTML)

@app.get("/api", summary="Root API status catalog")
def api_catalog():
    return {
        "status": "online",
        "service": "Employee Attrition & Retention Prediction Engine",
        "endpoints": {
            "ui": "/",
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health", summary="Health check")
def health_check():
    return {
        "status": "healthy" if predictor.is_loaded else "degraded",
        "model_loaded": predictor.is_loaded,
        "version": "1.0.0"
    }

@app.get("/model-info", summary="Model metadata")
def model_info():
    return {
        "model_type": "Random Forest Classifier Pipeline",
        "target_classes": ["Low Risk", "Moderate Risk", "High Risk"],
        "input_features": [
            "age", "department", "job_role", "monthly_income",
            "total_working_years", "years_at_company", "years_since_last_promotion",
            "overtime", "distance_from_home_km", "job_satisfaction", "work_life_balance"
        ],
        "engineered_features": [
            "tenure_ratio", "promotion_stagnation_ratio", "income_per_exp_year", "wellbeing_index"
        ],
        "performance_benchmark": {
            "test_accuracy": "82.00%",
            "macro_f1": "0.7937",
            "roc_auc_ovr": "0.9427"
        }
    }

@app.post("/predict", response_model=EmployeePredictionResponse, summary="Predict attrition risk")
def predict(request: EmployeePredictionRequest):
    try:
        return predictor.predict(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

# =====================================================================
# 5. EMBEDDED WEB UI HTML
# =====================================================================

EMBEDDED_UI_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8" />\n  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />\n  <title>Employee Attrition & Retention Predictor</title>\n  \n  <!-- Premium Typography: Plus Jakarta Sans -->\n  <link rel="preconnect" href="https://fonts.googleapis.com" />\n  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />\n  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />\n  \n  <!-- Tailwind CSS CDN -->\n  <script src="https://cdn.tailwindcss.com"></script>\n  \n  <!-- Lucide Icons -->\n  <script src="https://unpkg.com/lucide@latest"></script>\n  \n  <!-- Canvas Confetti -->\n  <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>\n\n  <script>\n    tailwind.config = {\n      theme: {\n        extend: {\n          fontFamily: {\n            sans: [\'"Plus Jakarta Sans"\', \'-apple-system\', \'BlinkMacSystemFont\', \'sans-serif\'],\n          },\n          colors: {\n            brand: {\n              navy: \'#0f172a\',\n              dark: \'#1e3a8a\',       // Deep Corporate Navy\n              blue: \'#2563eb\',       // Primary HR Blue\n              lightBlue: \'#dbeafe\',\n              light: \'#f8fafc\',      // Soft background\n              border: \'#e2e8f0\',     // Light card border\n              muted: \'#64748b\',      // Subdued text\n            },\n            risk: {\n              lowBg: \'#f0fdf4\',\n              lowBorder: \'#bbf7d0\',\n              lowText: \'#166534\',\n              lowIcon: \'#16a34a\',\n              \n              modBg: \'#fffbeb\',\n              modBorder: \'#fde68a\',\n              modText: \'#92400e\',\n              modIcon: \'#d97706\',\n              \n              highBg: \'#fef2f2\',\n              highBorder: \'#fecaca\',\n              highText: \'#991b1b\',\n              highIcon: \'#dc2626\',\n            }\n          },\n          boxShadow: {\n            \'phone\': \'0 25px 60px -15px rgba(15, 23, 42, 0.4), 0 0 0 1px rgba(255,255,255,0.1), 0 0 0 8px #0f172a\',\n            \'subtle\': \'0 2px 10px -2px rgba(15, 23, 42, 0.06)\',\n            \'pill\': \'0 4px 14px -2px rgba(37, 99, 235, 0.35)\',\n          }\n        }\n      }\n    };\n  </script>\n\n  <style>\n    /* Mobile frame & app screen transitions */\n    .app-screen {\n      position: absolute;\n      inset: 0;\n      width: 100%;\n      height: 100%;\n      transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.4s cubic-bezier(0.16, 1, 0.3, 1);\n      display: flex;\n      flex-direction: column;\n      background: #f8fafc;\n      overflow-y: auto;\n      overflow-x: hidden;\n      -webkit-overflow-scrolling: touch;\n    }\n    .app-screen::-webkit-scrollbar { display: none; }\n    .screen-prev { transform: translateX(-100%); opacity: 0; pointer-events: none; }\n    .screen-active { transform: translateX(0); opacity: 1; pointer-events: auto; }\n    .screen-next { transform: translateX(100%); opacity: 0; pointer-events: none; }\n\n    /* Custom form styling */\n    .app-input-row {\n      display: flex;\n      align-items: center;\n      gap: 0.75rem;\n      padding: 0.65rem 0.85rem;\n      background: #ffffff;\n      border: 1px solid #e2e8f0;\n      border-radius: 1rem;\n      transition: border-color 0.2s, box-shadow 0.2s;\n    }\n    .app-input-row:focus-within {\n      border-color: #2563eb;\n      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);\n    }\n    .icon-bubble {\n      width: 2rem;\n      height: 2rem;\n      border-radius: 9999px;\n      background: #eff6ff;\n      color: #2563eb;\n      display: flex;\n      align-items: center;\n      justify-content: center;\n      flex-shrink: 0;\n    }\n\n    /* Rating buttons */\n    .rating-pill {\n      flex: 1;\n      padding: 0.45rem 0;\n      font-size: 0.75rem;\n      font-weight: 700;\n      border-radius: 0.6rem;\n      border: 1px solid #e2e8f0;\n      background: #ffffff;\n      color: #64748b;\n      text-align: center;\n      cursor: pointer;\n      transition: all 0.2s;\n    }\n    .rating-pill.active {\n      background: #1e3a8a;\n      color: #ffffff;\n      border-color: #1e3a8a;\n      box-shadow: 0 2px 8px -1px rgba(30, 58, 138, 0.3);\n    }\n\n    /* Scenario pill buttons */\n    .scenario-pill {\n      font-size: 0.68rem;\n      font-weight: 700;\n      padding: 0.35rem 0.75rem;\n      border-radius: 9999px;\n      border: 1px solid #e2e8f0;\n      background: #ffffff;\n      color: #1e3a8a;\n      cursor: pointer;\n      transition: all 0.2s;\n      white-space: nowrap;\n    }\n    .scenario-pill:hover {\n      background: #eff6ff;\n      border-color: #93c5fd;\n    }\n    .scenario-pill:active { transform: scale(0.97); }\n\n    .btn-primary {\n      background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);\n      color: #ffffff;\n      transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;\n    }\n    .btn-primary:active { transform: scale(0.98); }\n  </style>\n</head>\n<body class="bg-[#0b1329] text-slate-900 min-h-screen flex flex-col items-center justify-start p-2 sm:p-6 antialiased selection:bg-blue-500 selection:text-white">\n\n  <!-- TOP BAR & CONTROLS -->\n  <header class="w-full max-w-5xl flex flex-wrap items-center justify-between gap-4 py-3 px-4 mb-4 bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 text-white shadow-lg">\n    <div class="flex items-center gap-3">\n      <div class="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center text-white font-black shadow-md">\n        <i data-lucide="users" class="w-5 h-5"></i>\n      </div>\n      <div>\n        <h1 class="text-sm font-bold tracking-tight text-white flex items-center gap-2">\n          Employee Attrition Predictor\n          <span class="text-[10px] uppercase font-extrabold px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">AI ML System</span>\n        </h1>\n        <p class="text-[11px] text-slate-400">Enterprise Workforce Retention Intelligence</p>\n      </div>\n    </div>\n\n    <!-- Quick Scenario Presets -->\n    <div class="flex items-center gap-2 overflow-x-auto py-1">\n      <span class="text-xs font-semibold text-slate-400 mr-1 hidden sm:inline">Presets:</span>\n      <button onclick="setScenario(\'low\')" class="scenario-pill bg-emerald-950/40 text-emerald-300 border-emerald-800/60 hover:bg-emerald-900/60">\n        🌱 Low Risk (Engaged)\n      </button>\n      <button onclick="setScenario(\'moderate\')" class="scenario-pill bg-amber-950/40 text-amber-300 border-amber-800/60 hover:bg-amber-900/60">\n        ⚠️ Moderate Risk (Watch)\n      </button>\n      <button onclick="setScenario(\'high\')" class="scenario-pill bg-rose-950/40 text-rose-300 border-rose-800/60 hover:bg-rose-900/60">\n        🚨 High Risk (Burnout)\n      </button>\n    </div>\n  </header>\n\n  <!-- MAIN PHONE MOCKUP CONTAINER -->\n  <main class="w-full flex justify-center py-2">\n    <div class="relative w-[385px] h-[780px] max-w-full bg-slate-900 rounded-[50px] p-3 shadow-phone border border-slate-800">\n      \n      <!-- Inner Screen Bezel -->\n      <div class="relative w-full h-full bg-[#f8fafc] rounded-[42px] overflow-hidden flex flex-col border border-slate-200">\n        \n        <!-- Dynamic Island / Speaker Notch -->\n        <div class="absolute top-2.5 left-1/2 -translate-x-1/2 w-28 h-6 bg-slate-950 rounded-full z-50 flex items-center justify-between px-2.5">\n          <div class="w-2.5 h-2.5 rounded-full bg-slate-900"></div>\n          <div class="w-2 h-2 rounded-full bg-blue-900/40"></div>\n        </div>\n\n        <!-- APP SCREENS CONTAINER -->\n        <div class="relative w-full h-full overflow-hidden">\n          \n          <!-- ============================================================ -->\n          <!-- SCREEN 1: WELCOME / INTRO SCREEN -->\n          <!-- ============================================================ -->\n          <section id="scr1" class="app-screen screen-active p-6 pt-11 justify-between">\n            <div class="flex flex-col items-center text-center">\n              \n              <!-- Brand Icon -->\n              <div class="w-14 h-14 rounded-2xl bg-gradient-to-tr from-blue-700 to-indigo-600 flex items-center justify-center text-white shadow-md mb-3 mt-1">\n                <i data-lucide="shield-alert" class="w-7 h-7"></i>\n              </div>\n\n              <h2 class="text-2xl font-extrabold text-slate-900 tracking-tight leading-tight">\n                Employee Attrition<br/><span class="text-blue-600">Prediction Engine</span>\n              </h2>\n              <p class="text-xs text-slate-500 font-medium mt-1.5 max-w-[260px]">\n                Identify flight risks, detect burnout drivers, and safeguard top corporate talent before departure.\n              </p>\n\n              <!-- Central SVG Illustration -->\n              <div class="my-5 w-56 h-48 relative flex items-center justify-center">\n                <svg viewBox="0 0 240 200" class="w-full h-full drop-shadow-sm">\n                  <!-- Base decorative circles -->\n                  <circle cx="120" cy="100" r="85" fill="#eff6ff" />\n                  <circle cx="120" cy="100" r="68" fill="#dbeafe" opacity="0.6" />\n                  \n                  <!-- Analytics Dashboard Cards in SVG -->\n                  <rect x="50" y="45" width="140" height="95" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="2" />\n                  <rect x="62" y="58" width="55" height="8" rx="4" fill="#3b82f6" />\n                  <rect x="62" y="72" width="116" height="4" rx="2" fill="#e2e8f0" />\n                  <rect x="62" y="80" width="85" height="4" rx="2" fill="#e2e8f0" />\n                  \n                  <!-- Mini Chart in Card -->\n                  <path d="M 65 118 L 85 105 L 105 112 L 125 90 L 145 95 L 165 75" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" />\n                  <circle cx="125" cy="90" r="4" fill="#2563eb" />\n                  <circle cx="165" cy="75" r="4.5" fill="#dc2626" />\n\n                  <!-- Overlapping Employee Avatars -->\n                  <circle cx="90" cy="155" r="22" fill="#1e3a8a" />\n                  <circle cx="90" cy="148" r="8" fill="#ffffff" />\n                  <path d="M76 168 C76 158, 104 158, 104 168 Z" fill="#ffffff" />\n\n                  <circle cx="150" cy="155" r="22" fill="#2563eb" />\n                  <circle cx="150" cy="148" r="8" fill="#ffffff" />\n                  <path d="M136 168 C136 158, 164 158, 164 168 Z" fill="#ffffff" />\n\n                  <!-- Badge -->\n                  <rect x="95" y="132" width="50" height="18" rx="9" fill="#10b981" />\n                  <text x="120" y="144" font-size="8.5" font-weight="bold" fill="#ffffff" text-anchor="middle">RETAIN</text>\n                </svg>\n              </div>\n\n              <!-- Value Props Pills -->\n              <div class="grid grid-cols-3 gap-2 w-full text-center">\n                <div class="p-2.5 bg-white rounded-xl border border-slate-200 shadow-subtle">\n                  <i data-lucide="activity" class="w-4 h-4 text-blue-600 mx-auto mb-1"></i>\n                  <span class="text-[10px] font-bold text-slate-700 block">Flight Risk Meter</span>\n                </div>\n                <div class="p-2.5 bg-white rounded-xl border border-slate-200 shadow-subtle">\n                  <i data-lucide="flame" class="w-4 h-4 text-amber-500 mx-auto mb-1"></i>\n                  <span class="text-[10px] font-bold text-slate-700 block">Burnout Alert</span>\n                </div>\n                <div class="p-2.5 bg-white rounded-xl border border-slate-200 shadow-subtle">\n                  <i data-lucide="sparkles" class="w-4 h-4 text-emerald-600 mx-auto mb-1"></i>\n                  <span class="text-[10px] font-bold text-slate-700 block">Retention Tips</span>\n                </div>\n              </div>\n\n            </div>\n\n            <!-- Start Assessment CTA -->\n            <div class="w-full mt-3">\n              <button onclick="navTo(2)" class="w-full btn-primary py-3.5 px-6 rounded-full font-bold text-sm flex items-center justify-center gap-2 shadow-pill">\n                <span>Start Employee Assessment</span>\n                <i data-lucide="arrow-right" class="w-4 h-4"></i>\n              </button>\n              <div class="w-24 h-1 bg-slate-300 rounded-full mx-auto mt-3 opacity-60"></div>\n            </div>\n          </section>\n\n\n          <!-- ============================================================ -->\n          <!-- SCREEN 2: INTERACTIVE ASSESSMENT FORM -->\n          <!-- ============================================================ -->\n          <section id="scr2" class="app-screen screen-next p-5 pt-10 justify-between">\n            <div>\n              <!-- Header Bar -->\n              <div class="flex items-center justify-between mb-2">\n                <button onclick="navTo(1)" class="w-8 h-8 rounded-full bg-white border border-slate-200 text-slate-700 flex items-center justify-center hover:bg-slate-100 transition shadow-subtle">\n                  <i data-lucide="arrow-left" class="w-4 h-4"></i>\n                </button>\n                <h2 class="text-sm font-bold text-slate-900 tracking-tight">Workplace Parameters</h2>\n                <div class="w-8"></div>\n              </div>\n\n              <!-- Input Form Fields -->\n              <div class="space-y-2.5 text-left">\n                \n                <!-- 1. Department -->\n                <div class="app-input-row">\n                  <div class="icon-bubble"><i data-lucide="building" class="w-4 h-4"></i></div>\n                  <div class="flex-1 pr-1">\n                    <label for="formDept" class="block text-[10px] font-bold text-brand-muted uppercase tracking-wider">Department</label>\n                    <select id="formDept" onchange="updateRoles()" class="w-full bg-transparent text-xs font-bold text-slate-900 focus:outline-none cursor-pointer">\n                      <option value="Technology" selected>Technology</option>\n                      <option value="Sales & Marketing">Sales & Marketing</option>\n                      <option value="Finance & Operations">Finance & Operations</option>\n                      <option value="Human Resources">Human Resources</option>\n                      <option value="Healthcare & R&D">Healthcare & R&D</option>\n                    </select>\n                  </div>\n                </div>\n\n                <!-- 2. Job Role -->\n                <div class="app-input-row">\n                  <div class="icon-bubble"><i data-lucide="briefcase" class="w-4 h-4"></i></div>\n                  <div class="flex-1 pr-1">\n                    <label for="formRole" class="block text-[10px] font-bold text-brand-muted uppercase tracking-wider">Job Role</label>\n                    <select id="formRole" class="w-full bg-transparent text-xs font-bold text-slate-900 focus:outline-none cursor-pointer">\n                      <!-- Filled dynamically by updateRoles() -->\n                    </select>\n                  </div>\n                </div>\n\n                <!-- 3. Age & Monthly Income (Grid) -->\n                <div class="grid grid-cols-2 gap-2">\n                  <div class="app-input-row">\n                    <div class="flex-1">\n                      <label for="formAge" class="block text-[10px] font-bold text-brand-muted uppercase">Age</label>\n                      <input type="number" id="formAge" value="34" min="18" max="75" class="w-full bg-transparent text-xs font-bold text-slate-900 focus:outline-none" />\n                    </div>\n                    <span class="text-[10px] font-semibold text-slate-400">yrs</span>\n                  </div>\n\n                  <div class="app-input-row">\n                    <div class="flex-1">\n                      <label for="formIncome" class="block text-[10px] font-bold text-brand-muted uppercase">Monthly CTC</label>\n                      <div class="flex items-center">\n                        <span class="text-[10px] font-bold text-slate-400 mr-0.5">$</span>\n                        <input type="number" id="formIncome" value="8500" step="100" min="1000" class="w-full bg-transparent text-xs font-bold text-slate-900 focus:outline-none" />\n                      </div>\n                    </div>\n                  </div>\n                </div>\n\n                <!-- 4. Experience & Tenure (Grid) -->\n                <div class="grid grid-cols-3 gap-2">\n                  <div class="app-input-row px-2">\n                    <div class="flex-1">\n                      <label for="formTotExp" class="block text-[9px] font-bold text-brand-muted uppercase">Total Exp</label>\n                      <input type="number" id="formTotExp" value="10" min="0" max="50" class="w-full bg-transparent text-xs font-bold text-slate-900 focus:outline-none" />\n                    </div>\n                    <span class="text-[9px] text-slate-400">y</span>\n                  </div>\n\n                  <div class="app-input-row px-2">\n                    <div class="flex-1">\n                      <label for="formTenure" class="block text-[9px] font-bold text-brand-muted uppercase">Tenure</label>\n                      <input type="number" id="formTenure" value="4" min="0" max="40" class="w-full bg-transparent text-xs font-bold text-slate-900 focus:outline-none" />\n                    </div>\n                    <span class="text-[9px] text-slate-400">y</span>\n                  </div>\n\n                  <div class="app-input-row px-2">\n                    <div class="flex-1">\n                      <label for="formPromote" class="block text-[9px] font-bold text-brand-muted uppercase">Promoted</label>\n                      <input type="number" id="formPromote" value="1" min="0" max="30" class="w-full bg-transparent text-xs font-bold text-slate-900 focus:outline-none" />\n                    </div>\n                    <span class="text-[9px] text-slate-400">y ago</span>\n                  </div>\n                </div>\n\n                <!-- 5. Overtime & Commute Distance -->\n                <div class="grid grid-cols-2 gap-2">\n                  <div class="app-input-row">\n                    <div class="flex-1">\n                      <label for="formOvertime" class="block text-[10px] font-bold text-brand-muted uppercase">Overtime</label>\n                      <select id="formOvertime" class="w-full bg-transparent text-xs font-bold text-slate-900 focus:outline-none cursor-pointer">\n                        <option value="No" selected>No (Normal)</option>\n                        <option value="Yes">Yes (Frequent)</option>\n                      </select>\n                    </div>\n                  </div>\n\n                  <div class="app-input-row">\n                    <div class="flex-1">\n                      <label for="formDistance" class="block text-[10px] font-bold text-brand-muted uppercase">Commute</label>\n                      <input type="number" id="formDistance" value="12" min="1" max="100" class="w-full bg-transparent text-xs font-bold text-slate-900 focus:outline-none" />\n                    </div>\n                    <span class="text-[10px] font-semibold text-slate-400">km</span>\n                  </div>\n                </div>\n\n                <!-- 6. Job Satisfaction (1 to 4) -->\n                <div class="bg-white p-2.5 rounded-xl border border-slate-200">\n                  <div class="flex justify-between items-center mb-1">\n                    <span class="text-[10px] font-bold text-brand-muted uppercase">Job Satisfaction</span>\n                    <span id="txtSatLabel" class="text-[10px] font-bold text-blue-600">High (3)</span>\n                  </div>\n                  <div class="flex gap-1.5" id="groupSat">\n                    <button type="button" onclick="setRating(\'Sat\', 1, \'Low\')" class="rating-pill">1 Low</button>\n                    <button type="button" onclick="setRating(\'Sat\', 2, \'Med\')" class="rating-pill">2 Med</button>\n                    <button type="button" onclick="setRating(\'Sat\', 3, \'High\')" class="rating-pill active">3 High</button>\n                    <button type="button" onclick="setRating(\'Sat\', 4, \'Very High\')" class="rating-pill">4 V.High</button>\n                  </div>\n                </div>\n\n                <!-- 7. Work-Life Balance (1 to 4) -->\n                <div class="bg-white p-2.5 rounded-xl border border-slate-200">\n                  <div class="flex justify-between items-center mb-1">\n                    <span class="text-[10px] font-bold text-brand-muted uppercase">Work-Life Balance</span>\n                    <span id="txtWlbLabel" class="text-[10px] font-bold text-blue-600">Good (3)</span>\n                  </div>\n                  <div class="flex gap-1.5" id="groupWlb">\n                    <button type="button" onclick="setRating(\'Wlb\', 1, \'Bad\')" class="rating-pill">1 Bad</button>\n                    <button type="button" onclick="setRating(\'Wlb\', 2, \'Average\')" class="rating-pill">2 Avg</button>\n                    <button type="button" onclick="setRating(\'Wlb\', 3, \'Good\')" class="rating-pill active">3 Good</button>\n                    <button type="button" onclick="setRating(\'Wlb\', 4, \'Excellent\')" class="rating-pill">4 Best</button>\n                  </div>\n                </div>\n\n              </div>\n            </div>\n\n            <!-- Predict CTA Button -->\n            <div class="mt-3">\n              <button id="btnPredict" onclick="executePrediction()" class="w-full btn-primary py-3.5 px-6 rounded-full font-bold text-sm flex items-center justify-center gap-2 shadow-pill">\n                <i data-lucide="sparkles" class="w-4 h-4 text-blue-200"></i>\n                <span id="txtPredict">Predict Attrition Risk</span>\n              </button>\n              <div class="w-24 h-1 bg-slate-300 rounded-full mx-auto mt-2.5 opacity-60"></div>\n            </div>\n          </section>\n\n\n          <!-- ============================================================ -->\n          <!-- SCREEN 3: PREDICTION RESULT & RETENTION PLAN -->\n          <!-- ============================================================ -->\n          <section id="scr3" class="app-screen screen-next p-5 pt-10 justify-between">\n            <div>\n              <!-- Header Bar -->\n              <div class="flex items-center justify-between mb-2">\n                <button onclick="navTo(2)" class="w-8 h-8 rounded-full bg-white border border-slate-200 text-slate-700 flex items-center justify-center hover:bg-slate-100 transition shadow-subtle">\n                  <i data-lucide="arrow-left" class="w-4 h-4"></i>\n                </button>\n                <h2 class="text-sm font-bold text-slate-900 tracking-tight">Assessment Output</h2>\n                <div class="w-8"></div>\n              </div>\n\n              <!-- DYNAMIC RESULT CARD -->\n              <div id="cardResult" class="bg-risk-lowBg border border-risk-lowBorder rounded-2xl p-4 text-center mb-2.5 shadow-subtle transition-all duration-300">\n                \n                <!-- Status Icon Badge -->\n                <div id="badgeResultIcon" class="w-10 h-10 rounded-full bg-risk-lowIcon text-white flex items-center justify-center mx-auto mb-2 shadow-md">\n                  <i id="iconResult" data-lucide="check-circle-2" class="w-5 h-5"></i>\n                </div>\n\n                <!-- Risk Level Headline -->\n                <h3 id="txtResultTitle" class="text-xl font-extrabold text-risk-lowText tracking-tight">\n                  Low Risk\n                </h3>\n\n                <!-- Probability Subtitle -->\n                <p class="text-xs font-semibold text-slate-600 mt-0.5">\n                  Flight Probability: <span id="txtFlightProb" class="font-bold text-slate-900">18.4%</span>\n                </p>\n\n                <!-- Action Tag Badge -->\n                <div class="mt-2 inline-block">\n                  <span id="tagStatus" class="px-3 py-1 rounded-full text-[10.5px] font-bold bg-white/80 border border-slate-200 text-slate-800 shadow-xs">\n                    Thriving Talent 🌱\n                  </span>\n                </div>\n              </div>\n\n              <!-- KEY RISK DRIVERS CARD -->\n              <div class="bg-white rounded-xl p-3 border border-slate-200 shadow-subtle mb-2">\n                <h4 class="text-[11px] font-bold text-slate-900 mb-1.5 flex items-center gap-1.5">\n                  <i data-lucide="compass" class="w-3.5 h-3.5 text-blue-600"></i>\n                  <span>Key Influencing Signals</span>\n                </h4>\n                <ul id="listDrivers" class="space-y-1 text-[11px] text-slate-600">\n                  <li class="flex items-center gap-1.5">✓ Well-calibrated compensation for tenure</li>\n                  <li class="flex items-center gap-1.5">✓ Standard workload with no overtime pressure</li>\n                </ul>\n              </div>\n\n              <!-- HR RECOMMENDATIONS CARD -->\n              <div class="bg-white rounded-xl p-3 border border-slate-200 shadow-subtle mb-2">\n                <h4 class="text-[11px] font-bold text-slate-900 mb-1.5 flex items-center gap-1.5">\n                  <i data-lucide="lightbulb" class="w-3.5 h-3.5 text-amber-500"></i>\n                  <span>Retention Recommendations</span>\n                </h4>\n                <ul id="listRecommendations" class="space-y-1 text-[11px] text-slate-600">\n                  <li>• Maintain quarterly career velocity discussions.</li>\n                  <li>• Consider mentorship leadership opportunities.</li>\n                </ul>\n              </div>\n\n              <!-- SUMMARY CARD -->\n              <div class="bg-white rounded-xl p-3 border border-slate-200 shadow-subtle text-[11px]">\n                <h4 class="text-[11px] font-bold text-slate-900 mb-1.5 pb-1 border-b border-slate-100 flex justify-between">\n                  <span>Profile Summary</span>\n                  <span id="resRoleDept" class="text-brand-muted font-medium">Software Engineer • Tech</span>\n                </h4>\n                <div class="grid grid-cols-2 gap-x-2 gap-y-1 text-slate-600">\n                  <div>Income: <span id="resIncome" class="font-bold text-slate-900">$8,500/mo</span></div>\n                  <div>Tenure: <span id="resTenure" class="font-bold text-slate-900">4 yrs</span></div>\n                  <div>Overtime: <span id="resOvertime" class="font-bold text-slate-900">No</span></div>\n                  <div>Satisfaction: <span id="resSat" class="font-bold text-slate-900">3 / 4</span></div>\n                </div>\n              </div>\n\n            </div>\n\n            <!-- Bottom CTA: Assess Again -->\n            <div class="mt-2">\n              <button onclick="navTo(2)" class="w-full btn-primary py-3 px-6 rounded-full font-bold text-xs flex items-center justify-center gap-2 shadow-pill mb-2">\n                <i data-lucide="rotate-cw" class="w-3.5 h-3.5"></i>\n                <span>Assess Another Profile</span>\n              </button>\n\n              <p class="text-center text-[10px] text-slate-400 font-semibold">\n                Proactive retention builds lasting organizational excellence.\n              </p>\n              <div class="w-24 h-1 bg-slate-300 rounded-full mx-auto mt-2 opacity-60"></div>\n            </div>\n          </section>\n\n        </div>\n      </div>\n    </div>\n  </main>\n\n  <!-- JAVASCRIPT APP LOGIC -->\n  <script>\n    // Department to Role Mapping\n    const DEPT_ROLES = {\n      "Technology": ["Software Engineer", "DevOps Specialist", "Data Scientist", "Engineering Manager"],\n      "Sales & Marketing": ["Sales Executive", "Account Manager", "Marketing Strategist", "Business Dev Rep"],\n      "Finance & Operations": ["Financial Analyst", "Operations Specialist", "Auditor", "Finance Director"],\n      "Human Resources": ["HR Specialist", "Talent Acquisition Lead", "HR Business Partner"],\n      "Healthcare & R&D": ["Research Scientist", "Clinical Specialist", "Lab Technician", "R&D Director"]\n    };\n\n    let selectedSat = 3;\n    let selectedWlb = 3;\n\n    function updateRoles() {\n      const dept = document.getElementById(\'formDept\').value;\n      const roleSelect = document.getElementById(\'formRole\');\n      roleSelect.innerHTML = \'\';\n      const roles = DEPT_ROLES[dept] || ["General Specialist"];\n      roles.forEach((r, idx) => {\n        const opt = document.createElement(\'option\');\n        opt.value = r;\n        opt.textContent = r;\n        if (idx === 0) opt.selected = true;\n        roleSelect.appendChild(opt);\n      });\n    }\n\n    function setRating(type, val, label) {\n      if (type === \'Sat\') {\n        selectedSat = val;\n        document.getElementById(\'txtSatLabel\').textContent = `${label} (${val})`;\n        updateRatingPills(\'groupSat\', val);\n      } else {\n        selectedWlb = val;\n        document.getElementById(\'txtWlbLabel\').textContent = `${label} (${val})`;\n        updateRatingPills(\'groupWlb\', val);\n      }\n    }\n\n    function updateRatingPills(groupId, activeVal) {\n      const buttons = document.querySelectorAll(`#${groupId} .rating-pill`);\n      buttons.forEach((btn, idx) => {\n        if (idx + 1 === activeVal) {\n          btn.classList.add(\'active\');\n        } else {\n          btn.classList.remove(\'active\');\n        }\n      });\n    }\n\n    // Screen Navigation (1, 2, 3)\n    function navTo(screenNum) {\n      [1, 2, 3].forEach(idx => {\n        const el = document.getElementById(`scr${idx}`);\n        if (!el) return;\n        el.className = \'app-screen\';\n        if (idx === screenNum) {\n          el.classList.add(\'screen-active\');\n        } else if (idx < screenNum) {\n          el.classList.add(\'screen-prev\');\n        } else {\n          el.classList.add(\'screen-next\');\n        }\n      });\n      const activeEl = document.getElementById(`scr${screenNum}`);\n      if (activeEl) activeEl.scrollTop = 0;\n    }\n\n    // Prediction Execution with smooth API call\n    async function executePrediction() {\n      const btn = document.getElementById(\'btnPredict\');\n      const txt = document.getElementById(\'txtPredict\');\n      btn.disabled = true;\n      txt.textContent = \'Analyzing Flight Risk...\';\n\n      const payload = {\n        age: parseInt(document.getElementById(\'formAge\').value) || 30,\n        department: document.getElementById(\'formDept\').value,\n        job_role: document.getElementById(\'formRole\').value,\n        monthly_income: parseFloat(document.getElementById(\'formIncome\').value) || 5000,\n        total_working_years: parseInt(document.getElementById(\'formTotExp\').value) || 5,\n        years_at_company: parseInt(document.getElementById(\'formTenure\').value) || 2,\n        years_since_last_promotion: parseInt(document.getElementById(\'formPromote\').value) || 0,\n        overtime: document.getElementById(\'formOvertime\').value,\n        distance_from_home_km: parseFloat(document.getElementById(\'formDistance\').value) || 10,\n        job_satisfaction: selectedSat,\n        work_life_balance: selectedWlb\n      };\n\n      try {\n        const apiUrl = window.location.origin && window.location.origin.startsWith(\'http\') ? \'/predict\' : \'http://127.0.0.1:8000/predict\';\n        const resp = await fetch(apiUrl, {\n          method: \'POST\',\n          headers: { \'Content-Type\': \'application/json\' },\n          body: JSON.stringify(payload)\n        });\n\n        if (!resp.ok) {\n          throw new Error(`Server returned HTTP ${resp.status}`);\n        }\n\n        const data = await resp.json();\n        renderResult(data, payload);\n      } catch (err) {\n        console.warn(\'API unavailable, running offline heuristic model:\', err);\n        fallbackPrediction(payload);\n      } finally {\n        navTo(3);\n        btn.disabled = false;\n        txt.textContent = \'Predict Attrition Risk\';\n      }\n    }\n\n    function renderResult(data, payload) {\n      const card = document.getElementById(\'cardResult\');\n      const title = document.getElementById(\'txtResultTitle\');\n      const probTxt = document.getElementById(\'txtFlightProb\');\n      const icon = document.getElementById(\'iconResult\');\n      const iconBadge = document.getElementById(\'badgeResultIcon\');\n      const tag = document.getElementById(\'tagStatus\');\n\n      const riskLevel = data.predicted_risk_level;\n      const probPct = data.risk_score_percentage ? `${data.risk_score_percentage}%` : `${Math.round(data.confidence * 100)}%`;\n      probTxt.textContent = probPct;\n\n      if (riskLevel.includes(\'Low\')) {\n        card.className = \'bg-risk-lowBg border border-risk-lowBorder rounded-2xl p-4 text-center mb-2.5 shadow-subtle\';\n        title.className = \'text-xl font-extrabold text-risk-lowText tracking-tight\';\n        title.textContent = \'Low Attrition Risk\';\n        iconBadge.className = \'w-10 h-10 rounded-full bg-risk-lowIcon text-white flex items-center justify-center mx-auto mb-2 shadow-md\';\n        icon.setAttribute(\'data-lucide\', \'check-circle-2\');\n        tag.textContent = data.status_tag || \'Thriving Talent 🌱\';\n\n        if (typeof confetti === \'function\') {\n          confetti({ particleCount: 40, spread: 55, origin: { y: 0.65 }, colors: [\'#2563eb\', \'#10b981\', \'#3b82f6\'] });\n        }\n      } else if (riskLevel.includes(\'High\')) {\n        card.className = \'bg-risk-highBg border border-risk-highBorder rounded-2xl p-4 text-center mb-2.5 shadow-subtle\';\n        title.className = \'text-xl font-extrabold text-risk-highText tracking-tight\';\n        title.textContent = \'High Attrition Risk\';\n        iconBadge.className = \'w-10 h-10 rounded-full bg-risk-highIcon text-white flex items-center justify-center mx-auto mb-2 shadow-md\';\n        icon.setAttribute(\'data-lucide\', \'alert-triangle\');\n        tag.textContent = data.status_tag || \'Flight Risk 🚨\';\n      } else {\n        card.className = \'bg-risk-modBg border border-risk-modBorder rounded-2xl p-4 text-center mb-2.5 shadow-subtle\';\n        title.className = \'text-xl font-extrabold text-risk-modText tracking-tight\';\n        title.textContent = \'Moderate Attrition Risk\';\n        iconBadge.className = \'w-10 h-10 rounded-full bg-risk-modIcon text-white flex items-center justify-center mx-auto mb-2 shadow-md\';\n        icon.setAttribute(\'data-lucide\', \'info\');\n        tag.textContent = data.status_tag || \'Watch List ⚠️\';\n      }\n\n      // Drivers\n      const listDrivers = document.getElementById(\'listDrivers\');\n      listDrivers.innerHTML = \'\';\n      (data.key_drivers || []).forEach(d => {\n        const li = document.createElement(\'li\');\n        li.className = \'flex items-center gap-1.5\';\n        li.textContent = `• ${d}`;\n        listDrivers.appendChild(li);\n      });\n\n      // Recommendations\n      const listRec = document.getElementById(\'listRecommendations\');\n      listRec.innerHTML = \'\';\n      (data.retention_recommendations || []).forEach(r => {\n        const li = document.createElement(\'li\');\n        li.textContent = `• ${r}`;\n        listRec.appendChild(li);\n      });\n\n      // Summary\n      document.getElementById(\'resRoleDept\').textContent = `${payload.job_role} • ${payload.department}`;\n      document.getElementById(\'resIncome\').textContent = `$${payload.monthly_income.toLocaleString()}/mo`;\n      document.getElementById(\'resTenure\').textContent = `${payload.years_at_company} yrs`;\n      document.getElementById(\'resOvertime\').textContent = payload.overtime;\n      document.getElementById(\'resSat\').textContent = `${payload.job_satisfaction} / 4`;\n\n      lucide.createIcons();\n    }\n\n    function fallbackPrediction(p) {\n      let score = 0.25;\n      if (p.overtime === \'Yes\') score += 0.25;\n      if (p.job_satisfaction <= 2) score += 0.22;\n      if (p.work_life_balance <= 1) score += 0.20;\n      if (p.years_since_last_promotion >= 3) score += 0.15;\n      if (p.distance_from_home_km >= 25) score += 0.10;\n\n      score = Math.min(0.95, Math.max(0.08, score));\n      let level = \'Moderate Risk\';\n      let tag = \'Watch List ⚠️\';\n      let drivers = [\'Mild disengagement indicators\', \'Review workload margins\'];\n      let recs = [\'Conduct 1-on-1 check-in\', \'Explore internal skill progression\'];\n\n      if (score <= 0.35) {\n        level = \'Low Risk\';\n        tag = \'Thriving Talent 🌱\';\n        drivers = [\'Competitive role alignment\', \'Healthy personal-professional balance\'];\n        recs = [\'Maintain current recognition programs\', \'Consider senior mentorship roles\'];\n      } else if (score >= 0.65) {\n        level = \'High Risk\';\n        tag = \'Flight Risk 🚨\';\n        drivers = [\'Overtime fatigue detected\', \'Promotion stagnation or commute friction\'];\n        recs = [\'Urgent executive compensation review\', \'Implement work-life balance guardrails\'];\n      }\n\n      renderResult({\n        predicted_risk_level: level,\n        risk_score_percentage: Math.round(score * 100),\n        confidence: 0.88,\n        status_tag: tag,\n        key_drivers: drivers,\n        retention_recommendations: recs\n      }, p);\n    }\n\n    // Quick Presets\n    function setScenario(type) {\n      if (type === \'low\') {\n        document.getElementById(\'formDept\').value = \'Technology\';\n        updateRoles();\n        document.getElementById(\'formRole\').value = \'Software Engineer\';\n        document.getElementById(\'formAge\').value = \'35\';\n        document.getElementById(\'formIncome\').value = \'12000\';\n        document.getElementById(\'formTotExp\').value = \'11\';\n        document.getElementById(\'formTenure\').value = \'5\';\n        document.getElementById(\'formPromote\').value = \'1\';\n        document.getElementById(\'formOvertime\').value = \'No\';\n        document.getElementById(\'formDistance\').value = \'6\';\n        setRating(\'Sat\', 4, \'Very High\');\n        setRating(\'Wlb\', 3, \'Good\');\n      } else if (type === \'moderate\') {\n        document.getElementById(\'formDept\').value = \'Finance & Operations\';\n        updateRoles();\n        document.getElementById(\'formRole\').value = \'Financial Analyst\';\n        document.getElementById(\'formAge\').value = \'29\';\n        document.getElementById(\'formIncome\').value = \'5400\';\n        document.getElementById(\'formTotExp\').value = \'5\';\n        document.getElementById(\'formTenure\').value = \'3\';\n        document.getElementById(\'formPromote\').value = \'3\';\n        document.getElementById(\'formOvertime\').value = \'No\';\n        document.getElementById(\'formDistance\').value = \'24\';\n        setRating(\'Sat\', 2, \'Med\');\n        setRating(\'Wlb\', 2, \'Average\');\n      } else if (type === \'high\') {\n        document.getElementById(\'formDept\').value = \'Sales & Marketing\';\n        updateRoles();\n        document.getElementById(\'formRole\').value = \'Sales Executive\';\n        document.getElementById(\'formAge\').value = \'27\';\n        document.getElementById(\'formIncome\').value = \'4100\';\n        document.getElementById(\'formTotExp\').value = \'4\';\n        document.getElementById(\'formTenure\').value = \'2\';\n        document.getElementById(\'formPromote\').value = \'2\';\n        document.getElementById(\'formOvertime\').value = \'Yes\';\n        document.getElementById(\'formDistance\').value = \'35\';\n        setRating(\'Sat\', 1, \'Low\');\n        setRating(\'Wlb\', 1, \'Bad\');\n      }\n      navTo(2);\n    }\n\n    // Initial setup\n    updateRoles();\n    lucide.createIcons();\n  </script>\n</body>\n</html>\n'

# =====================================================================
# 6. CLI DIRECT EXECUTION WITH AUTO-BROWSER LAUNCH
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time

    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    print("=" * 65)
    print("  * Employee Attrition & Retention Prediction System")
    print("  * Interactive Web UI:     http://127.0.0.1:8000")
    print("  * API Documentation:     http://127.0.0.1:8000/docs")
    print("=" * 65)
    uvicorn.run(app, host="127.0.0.1", port=8000)
