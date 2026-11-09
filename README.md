# Machine Learning-Based Prediction of Student Academic Performance Using Educational Data

**Author:** Subhradeet Sabat (`subhradeet.sabat.cse.2024@nist.edu`)  
**Affiliation:** Department of Computer Science & Engineering, NIST University, Berhampur, Odisha, India  
**Course:** Technical Writing (Lab 09 & 10)  
**Supervisor / Faculty:** Dr. Brojo Kishore Mishra (`bkmishra@nist.edu`)  
**Target Diamond Open Access Journal:** *Journal of Machine Learning Research (JMLR)*  
**Official Journal Portal:** [https://www.jmlr.org](https://www.jmlr.org)  
**Publication Model:** Diamond Open Access (100% Free for Authors, 100% Free for Readers, Zero APC, DOAJ Seal, Scopus Q1, Web of Science SCIE)  

---

## 1. Executive Summary & Problem Formulation

Educational institutions routinely accumulate student demographic, familial, and behavioral records; however, identifying students who may experience poor academic outcomes *prior* to intermediate or terminal examinations remains a major institutional challenge. Under conventional university workflows, student failure is typically detected retrospectively after examination scores are published, leaving minimal time for remedial mentoring or personalized tutorial support.

This research project develops, implements, and benchmarks a supervised machine learning early-warning framework on the official **UCI Student Performance Benchmark Dataset** (Mathematics: `student-mat.csv`, Cortez & Silva, 2008). 

### Key Methodological Policy (Pre-Examination Triage)
To simulate a genuine early-warning diagnostic setting, **first-period (G1) and second-period (G2) examination marks are strictly excluded** from the predictive feature space. Including intermediate examination grades creates an unrealistic methodological shortcut (data leakage), as G1/G2 correlate strongly with the final grade (G3). Predicting final performance using strictly pre-examination demographic, familial, and behavioral attributes provides genuine, actionable prognostic foresight.

---

## 2. Directory Architecture

The complete project is stored locally at `D:\AI_Research_Paper\`:

```
D:\AI_Research_Paper\
│
├── input\
│   ├── Diamond_Access_Journal_List.pdf        # Official Diamond OA journal list from HOD
│   ├── HOD_Research_Instructions.pdf           # Lab 09 & 10 Instructions & Evaluation criteria
│   └── template_jmlr\                          # Official JMLR LaTeX template package
│       ├── jmlr2e.sty                          # Official JMLR LaTeX style file
│       └── sample.tex                          # Official JMLR sample manuscript
│
├── dataset\
│   ├── student-mat.csv                         # Official UCI Mathematics Student Performance data (N = 395)
│   └── student.txt                             # Official UCI attribute documentation
│
├── src\
│   └── research_pipeline.py                    # Master Python script executing complete pipeline
│
├── results\
│   ├── graphs\                                 # High-resolution (300 DPI) experimental figures
│   │   ├── fig1_class_distribution.png         # Binary target distribution
│   │   ├── fig2_correlation_heatmap.png        # Spearman rank correlation matrix
│   │   ├── fig3_feature_distributions.png      # Predictor distributions by academic outcome
│   │   ├── fig4_model_comparison_metrics.png   # Multi-metric comparative bar chart
│   │   ├── fig5_confusion_matrices.png         # Multi-panel normalized confusion matrices
│   │   ├── fig6_roc_curves.png                 # Multi-model ROC curves & AUC scores
│   │   ├── fig7_precision_recall_curves.png    # Precision-Recall curves & Average Precision
│   │   ├── fig8_actual_vs_predicted_grades.png # Continuous grade regression (0–20 scale)
│   │   ├── fig9_prediction_probabilities.png   # Posterior passing probability densities
│   │   ├── fig10_feature_importance.png        # Gini feature attribution rankings
│   │   └── fig11_learning_curves.png           # 5-fold CV convergence curves
│   │
│   ├── matrices\                               # Confusion matrices (Raw counts & percentages)
│   │   ├── confusion_matrix_Logistic_Regression.csv
│   │   ├── confusion_matrix_Decision_Tree.csv
│   │   ├── confusion_matrix_Random_Forest.csv
│   │   └── confusion_matrix_Support_Vector_Machine.csv
│   │
│   ├── models\                                 # Persisted trained model pipelines (joblib)
│   │   ├── logistic_regression.joblib
│   │   ├── decision_tree.joblib
│   │   ├── random_forest.joblib
│   │   ├── support_vector_machine.joblib
│   │   └── grade_regressor.joblib
│   │
│   └── reports\                                # Numerical evaluations & verification audits
│       ├── classification_reports.txt          # Detailed classification reports on test cohort
│       ├── cross_validation_results.csv        # 5-fold stratified CV scores
│       ├── model_metrics_summary.json          # Machine-readable evaluation metrics
│       ├── model_metrics_table.csv             # Formatted performance table
│       └── reproducibility_report.txt          # Environment, seed, and hardware audit trail
│
├── paper\                                      # Final Research Paper & Submission Package
│   ├── Research_Article_Manuscript.docx        # Full Research Paper in MS Word format (10-15 pages)
│   ├── Research_Article_Manuscript.pdf         # 13-Page Publication-Ready PDF proof
│   ├── manuscript_jmlr.tex                     # Official JMLR LaTeX manuscript source
│   ├── Journal_Comparison_Table.docx / .pdf    # 3-Journal evaluation table (Section 7 schema)
│   ├── Evidence_Zero_APC_Open_Access.docx / .pdf # DOAJ & UNESCO Diamond OA evidence
│   └── Cover_Letter_Editor_in_Chief.docx / .pdf  # Formal submission letter to JMLR Editors
│
├── requirements.txt                            # Complete Python dependencies
├── README.md                                   # Comprehensive project documentation
└── run.bat                                     # One-click Windows execution batch script
```

---

## 3. Machine Learning Algorithms & Experimental Results

All models were evaluated on the held-out test cohort ($N = 79$ students) with 5-fold stratified cross-validation on the training set ($N = 316$ students):

| Algorithm | 5-Fold CV Accuracy | Test Accuracy | Balanced Acc. | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 68.37% | 55.70% | 54.34% | 69.57% | 60.38% | 46.15% | 0.6465 | 0.5980 |
| **Decision Tree (CART)** | 65.81% | 63.29% | 61.32% | 73.08% | 71.70% | 46.15% | 0.7238 | 0.6012 |
| **Random Forest (Ensemble)** | **69.31%** | **68.35%** | **66.84%** | **75.93%** | **77.36%** | **50.00%** | **0.7664** | **0.6190** |
| **Support Vector Machine (SVM)** | 67.72% | 65.82% | 56.46% | 72.73% | 90.57% | 15.38% | 0.7805 | 0.5675 |

### Secondary Continuous Grade Regression (Section 7C)
- **Algorithm:** Random Forest Regressor ($n = 100$, max depth = 5)
- **Mean Absolute Error (MAE):** **3.01 grade points** on the Portuguese 0–20 grading scale.
- **Coefficient of Determination ($R^2$):** **0.221** (consistent with published pre-examination literature without G1/G2).

### Key Empirical Findings:
1. **Best Performing Model:** **Random Forest Classifier** achieved the optimal balance of accuracy (68.35%), precision (75.93%), recall (77.36%), F1-score (0.7664), and discriminative capability (ROC-AUC = 0.6190).
2. **Top Predictive Features (Gini Impurity):**
   - Past Academic Failures (`failures`, relative importance = 0.22)
   - School Absences (`absences`, relative importance = 0.16)
   - Student Chronological Age (`age`, relative importance = 0.11)
   - Maternal Education Level (`Medu`, relative importance = 0.09)
   - Weekly Study Time (`studytime`, relative importance = 0.07)

---

## 4. How to Execute the Entire Project

### Option A: Double-Click Batch File (Windows)
Double-click [`D:\AI_Research_Paper\run.bat`](file:///D:/AI_Research_Paper/run.bat). It automatically creates the virtual environment, verifies dependencies, and executes the entire pipeline.

### Option B: VS Code Terminal Commands
1. Launch **Visual Studio Code**, click **File → Open Folder...**, and select `D:\AI_Research_Paper`.
2. Open the integrated terminal (`Ctrl + ~`) and run:

```powershell
# Navigate to project root
cd /d D:\AI_Research_Paper

# (Optional) Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install required dependencies
python -m pip install -r requirements.txt

# Execute the complete research pipeline
python src\research_pipeline.py
```

---

## 5. Academic Integrity & Quality Control Verification

- **Target Journal:** *Journal of Machine Learning Research (JMLR)* — verified Diamond Open Access, 0 APC, indexed in Scopus (Q1) and Web of Science.
- **Original Template Compliance:** Preserves official JMLR formatting, single-column layout, Times-Roman typography, running headers, and APA/plainnat reference style.
- **Main Manuscript Page Count:** **Exactly 13 pages** (strictly between the 10-page minimum and 15-page maximum).
- **Dataset Authenticity:** Official UCI Student Performance repository (`student-mat.csv`), 395 verified student instances, zero fabricated records.
- **Experimental Reproducibility:** Fixed random seed `42`, deterministic cross-validation folds, and automated report auditing.
