# CareerCompass AI - A Career Guidance and Recommendation Platform

An offline, ML-powered career counselling app built with **Python + Streamlit**.
It predicts a suitable tech career, analyzes skill gaps, and recommends courses
and interview questions — all running locally with **no paid APIs**.

## Tech Stack

- **Frontend:** Streamlit
- **ML:** scikit-learn (TF-IDF + classifier pipeline)
- **NLP:** NLTK-style keyword matching + Regular Expressions
- **Database:** SQLite (built-in `sqlite3`)
- **Visualization:** Matplotlib
- **File handling:** Pandas, pdfplumber / PyPDF2

## Project Structure

```
CareerCounsellor/
├── app.py                     # Home page (Streamlit entry point)
├── pages/                     # Multi-page UI
│   ├── 1_Student_Profile.py   # Phase 8  - profile form
│   ├── 2_Resume_Upload.py     # Phase 9  - PDF resume parsing
│   ├── 3_Career_Prediction.py # Phases 10-13 - prediction + gap + recs
│   ├── 4_Career_Report.py     # Phase 15 - downloadable report
│   └── 5_History.py           # SQLite history viewer
├── models/
│   ├── train_model.py         # Phases 4-6 - train, evaluate, save
│   ├── career_model.pkl       # saved pipeline (generated)
│   └── confusion_matrix.png   # evaluation plot (generated)
├── database/
│   └── db.py                  # Phase 14 - SQLite persistence layer
├── datasets/
│   ├── career_dataset.csv     # Phase 2 - synthetic training data (2000 rows)
│   ├── courses.csv            # course recommendations
│   └── interview_questions.csv# interview questions
├── utils/
│   ├── config.py              # careers, required skills, paths
│   ├── generate_datasets.py   # Phase 2 - dataset generator
│   ├── data_cleaning.py       # Phase 3 - text normalization
│   ├── resume_parser.py       # Phase 9 - PDF parsing
│   ├── predictor.py           # Phase 10 - model inference
│   ├── skill_gap.py           # Phase 11 - skill gap analysis
│   ├── recommender.py         # Phases 12-13 - course/interview recs
│   └── report_generator.py    # Phase 15 - report builder
├── reports/                   # generated .txt reports land here
├── requirements.txt
└── README.md
```

## Setup (run locally)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) regenerate the synthetic training dataset
python -m utils.generate_datasets

# 4. Train the ML model (creates models/career_model.pkl)
python -m models.train_model

# 5. Launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501) and use the
sidebar pages in order: **Profile → Resume → Prediction → Report**.

## How the ML works

Each training row is a cleaned, comma-separated skill string plus a CGPA,
labelled with a career. A `ColumnTransformer` applies **TF-IDF** to the skills
(using a custom tokenizer so multi-word skills like "machine learning" stay
intact) and **StandardScaler** to the CGPA. Three classifiers — Logistic
Regression, Random Forest and SVM — are compared with 5-fold cross-validation,
and the best one is pickled. Because the synthetic careers have distinct skill
signatures, accuracy on this dataset is very high; on messier real data expect
lower, more realistic scores.

## Testing (Phase 16)

Every module has a `__main__` block you can run directly to smoke-test it:

```bash
python -m utils.data_cleaning      # cleaning examples
python -m utils.predictor          # sample prediction
python -m utils.skill_gap          # sample gap analysis
python -m utils.recommender        # sample recommendations
python -m utils.report_generator   # sample report
python -m database.db              # initialize the database
```

## Build Progress

- [x] Phase 1 — Folder Structure
- [x] Phase 2 — Dataset Creation
- [x] Phase 3 — Data Cleaning
- [x] Phase 4 — Model Training
- [x] Phase 5 — Model Evaluation
- [x] Phase 6 — Save Model
- [x] Phase 7 — Streamlit UI
- [x] Phase 8 — Student Profile
- [x] Phase 9 — Resume Parser
- [x] Phase 10 — Career Prediction
- [x] Phase 11 — Skill Gap Analysis
- [x] Phase 12 — Course Recommendation
- [x] Phase 13 — Interview Recommendation
- [x] Phase 14 — SQLite Integration
- [x] Phase 15 — Career Report
- [x] Phase 16 — Testing
- [x] Phase 17 — Documentation
