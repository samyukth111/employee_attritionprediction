# Employee Attrition & Retention Prediction System

A production-ready machine learning application, REST API, and interactive mobile decision-support system that predicts workforce turnover risk (**Low Risk**, **Moderate Risk**, **High Risk**) based on employee demographic, compensatory, and psychometric workplace attributes.

---

## 📌 1. Project Overview & Problem Statement

Workforce turnover presents substantial financial and organizational challenges across enterprises. According to the Society for Human Resource Management (SHRM) and Mercer Workforce Studies:
* Replacing a departing skilled knowledge worker costs between **33% and 150%** of their annual salary.
* Retrospective exit interviews provide data only after valuable organizational expertise has walked out the door.
* Key drivers of preventable turnover—such as chronic uncompensated overtime, stagnation in career promotion, commute friction, and low job sentiment—often exhibit non-linear compounding interactions.

### Objective
To engineer an end-to-end Machine Learning solution that proactively assesses employee attributes and provides:
1. **Early-Warning Attrition Classification**:
   * **Low Risk ($\le 33\%$)**: Engaged, stable employee with healthy career velocity.
   * **Moderate Risk ($34\% - 66\%$)**: Early disengagement or stagnation indicators; requires proactive check-ins.
   * **High Risk ($> 66\%$)**: Imminent flight risk requiring urgent executive retention intervention.
2. **Key Driver Identification**: Explains why an employee is at risk (e.g., overtime burnout, commute burden, 3+ years without promotion).
3. **Actionable HR Retention Levers**: Generates concrete guidance (e.g., compensation review, workload rebalancing, mentorship assignments).

---

## 📊 2. Dataset Description & Features

### Dataset Disclosure
* **Dataset File**: `dataset.csv`
* **Size**: 3,000 corporate employee records ($80/20$ stratified train/test split: $2,400$ training / $600$ test records).
* **Source & Methodology**: Statistically grounded benchmark dataset constructed using fixed seed `random_seed=42`, modeled after the IBM HR Analytics framework and Mercer Workforce studies.

### Feature Schema

| Feature Name | Data Type | Value Range | HR Operational Definition |
| :--- | :---: | :---: | :--- |
| `age` | Numerical (int) | $21 - 62\text{ years}$ | Chronological employee age |
| `department` | Categorical | `Technology`, `Sales & Marketing`, `Finance & Operations`, `Human Resources`, `Healthcare & R&D` | Organizational functional business unit |
| `job_role` | Categorical | `Software Engineer`, `Sales Executive`, `Financial Analyst`, `HR Specialist`, `Research Scientist`, etc. | Specific job title and seniority level |
| `monthly_income` | Numerical (float) | $\$2,200 - \$24,000\text{ / month}$ | Gross monthly base compensation in USD |
| `total_working_years`| Numerical (int) | $0 - 40\text{ years}$ | Cumulative professional career experience |
| `years_at_company` | Numerical (int) | $0 - 30\text{ years}$ | Organizational tenure at current enterprise |
| `years_since_last_promotion`| Numerical (int) | $0 - 15\text{ years}$ | Advancement velocity & stagnation index |
| `overtime` | Categorical | `Yes` / `No` | Regular expectation of uncompensated overtime |
| `distance_from_home_km` | Numerical (float) | $1.0 - 48.0\text{ km}$ | Commuting distance friction factor |
| `job_satisfaction` | Ordinal (int) | $1\text{ (Low) to } 4\text{ (Very High)}$ | Employee sentiment regarding role and team |
| `work_life_balance` | Ordinal (int) | $1\text{ (Bad) to } 4\text{ (Excellent)}$ | Personal-professional equilibrium rating |
| **`attrition_risk_level`** | **Target Category** | `Low Risk`, `Moderate Risk`, `High Risk` | **3-tier flight risk classification** |

---

## ⚙️ 3. Preprocessing & Domain Feature Engineering

1. **Domain Feature Engineering (`EmployeeFeatureEngineer`)**:
   * $\text{tenure\_ratio} = \frac{\text{years\_at\_company}}{\max(\text{total\_working\_years}, 1)}$ (Organizational loyalty vs. job-hopping tendency)
   * $\text{promotion\_stagnation\_ratio} = \frac{\text{years\_since\_last\_promotion}}{\max(\text{years\_at\_company}, 1)}$ (Career stagnation relative to tenure)
   * $\text{income\_per\_exp\_year} = \frac{\text{monthly\_income}}{\max(\text{total\_working\_years}, 1)}$ (Compensation fairness and equity index)
   * $\text{wellbeing\_index} = (\text{job\_satisfaction} \times 0.55) + (\text{work\_life\_balance} \times 0.45)$ (Holistic employee morale)
2. **Column Transformations**:
   * Categorical features (`department`, `job_role`, `overtime`) encoded with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   * Continuous numerical features standardized with `StandardScaler()`.
3. **Data Leakage Prevention**:
   * Post-event target values (`attrition_risk_score`, `attrition`) are strictly excluded from feature inputs.
   * Model parameters and scaling statistics are fitted strictly on the $2,400$ training samples.

---

## 🔬 4. Machine Learning Algorithms & Benchmark Comparison

Three classification algorithms were compared under **5-fold stratified cross-validation** on an $80/20$ stratified split ($2,400$ training / $600$ test records, `random_state=42`). (Linear and Logistic Regression were excluded per specifications).

| Model Algorithm | 5-Fold CV Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | ROC-AUC (OvR Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Random Forest (In `model.pkl`)** | **81.13%** ($\pm 0.015$) | **82.00%** | **0.8287** | **0.7756** | **0.7937** | **0.8188** | **0.9427** |
| **Decision Tree Classifier** | 77.46% ($\pm 0.024$) | 79.83% | 0.7978 | 0.7659 | 0.7783 | 0.7986 | 0.9109 |
| **K-Nearest Neighbors (KNN)** | 73.63% ($\pm 0.019$) | 75.83% | 0.7471 | 0.7050 | 0.7195 | 0.7545 | 0.8918 |

* **Selected Model**: **Random Forest Classifier** ($150$ estimators, `max_depth=11`, `min_samples_split=4`, `min_samples_leaf=2`) serialized in `model.pkl`.

---

## 🚀 5. Installation & Execution

### Prerequisites
* Python 3.9+ installed and added to PATH.

### 1. Clone & Setup Virtual Environment
```powershell
cd employee-attrition-prediction
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Application & REST API
```powershell
python app.py
```
* The web browser will automatically open to `http://127.0.0.1:8000` displaying the interactive 3-screen phone interface.
* Interactive API Documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

---

## 🧪 6. API Reference & Sample Payloads

### Endpoints
* `GET /`: Serves the interactive mobile Web UI (or JSON status catalog if `Accept: application/json`).
* `GET /health`: Health status and model loaded verification.
* `GET /model-info`: Active model metadata, target classes, and feature schema.
* `POST /predict`: Real-time employee turnover prediction endpoint.

### Sample Prediction Request (`POST /predict`):
```json
{
  "age": 35,
  "department": "Technology",
  "job_role": "Software Engineer",
  "monthly_income": 12000.0,
  "total_working_years": 11,
  "years_at_company": 5,
  "years_since_last_promotion": 1,
  "overtime": "No",
  "distance_from_home_km": 6.0,
  "job_satisfaction": 4,
  "work_life_balance": 3
}
```

### Sample Prediction Response:
```json
{
  "predicted_risk_level": "Low Risk",
  "risk_score_percentage": 15.9,
  "confidence": 0.895,
  "probabilities": {
    "Low Risk": 0.895,
    "Moderate Risk": 0.098,
    "High Risk": 0.007
  },
  "status_tag": "Thriving Talent 🌱",
  "key_drivers": [
    "Positive role alignment and competitive compensation",
    "Healthy personal-professional work-life equilibrium"
  ],
  "retention_recommendations": [
    "Maintain ongoing recognition and high-performance incentives.",
    "Engage in leadership pipeline and technical mentorship roles.",
    "Keep tracking annual satisfaction and compensation growth."
  ],
  "summary": {
    "age": "35",
    "department": "Technology",
    "job_role": "Software Engineer",
    "monthly_income": "$12,000",
    "tenure": "5 yrs",
    "overtime": "No",
    "satisfaction": "4 / 4"
  }
}
```

---

## 📁 7. Repository Directory Structure

```text
employee-attrition-prediction/
├── app.py              # Full FastAPI server + embedded interactive UI + auto-browser launcher
├── model.pkl           # Serialized Scikit-Learn Random Forest pipeline
├── dataset.csv         # 3,000-record benchmark workforce dataset
├── requirements.txt    # Production dependencies
├── README.md           # Project documentation and API reference
├── report.pdf          # 2-page academic assignment report with embedded charts
└── screenshots/
    └── app.png         # High-resolution screenshot of the running UI
```

---

## ⚠️ 8. Limitations & Real-World HR Recommendations

1. **Benchmark Simulation**: The dataset models generalized corporate turnover dynamics. Individual workplace dynamics can vary based on company culture, leadership quality, and economic cycles.
2. **Qualitative Factors**: Qualitative elements like interpersonal team friction or personal family circumstances are not fully captured by numerical metrics.
3. **Recommendation for Production**: Enterprises should integrate predictive analytics with quarterly pulse surveys and stay-interviews while adhering to labor privacy and algorithmic fairness regulations.
