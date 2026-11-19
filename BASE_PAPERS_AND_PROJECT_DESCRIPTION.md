# 📄 Academic Project Description & Base Paper Guide
**Project Title**: Machine Learning Pipeline for Imbalanced Credit Card Fraud Detection with Cost-Sensitive Learning & Threshold Optimization  
**Domain**: Financial Machine Learning / Data Mining / Cyber Security  
**Target Academic Degree**: M.Tech (Master of Technology) in Computer Science / AI & ML / Data Science  

---

## 1. Primary Base Papers (Foundational References)

These are the seminal, peer-reviewed papers upon which this exact dataset, methodology, and evaluation framework are grounded. You can cite these as your **Primary Base Papers** in your M.Tech project synopsis, thesis literature review, and presentation:

### Base Paper 1: The Dataset Benchmark Paper
* **Title**: *Calibrating Probability with Undersampling for Unbalanced Classification*
* **Authors**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi
* **Published In**: *IEEE Symposium Series on Computational Intelligence (SSCI)*, 2015
* **DOI**: `10.1109/SSCI.2015.33`
* **Why this is your Base Paper**: This paper is by the research group (Machine Learning Group - ULB) that created and published the `creditcard.csv` dataset. It establishes why precision-recall metrics and probability calibration are essential over standard accuracy in credit card fraud detection.

### Base Paper 2: The Core Temporal & Imbalance Methodology
* **Title**: *Learner-Consumer-Evaluator: A Design Pattern for Big Data Analytics and Fraud Detection*
* **Authors**: Andrea Dal Pozzolo, Olivier Caelen, Yann-Aël Le Borgne, Serge Waterschoot, Gianluca Bontempi
* **Published In**: *ACM Technological Innovations in Financial Services*, 2014
* **Key Concept Used**: Demonstrates why random K-Fold cross-validation leaks temporal patterns (data leakage), establishing the requirement for **chronological sorting and time-based train/val/test splits**.

### Base Paper 3: The State-of-the-Art Survey & Handbook Reference
* **Title**: *Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook*
* **Authors**: Yann-Aël Le Borgne, Gianluca Bontempi
* **Published In**: *Université Libre de Bruxelles (ULB)*, 2021 / Fraud Detection Handbook
* **Key Concept Used**: Cost-sensitive parameter scaling (`scale_pos_weight`, `class_weight='balanced_subsample'`), PR-AUC optimization, and why SMOTE should only be evaluated on training splits without test contamination.

---

## 2. Formal Academic Project Description (Abstract & Summary)

You can copy and adapt this description directly into your **Project Report / Synopsis / Thesis Abstract**:

### Abstract
> Credit card fraud presents a critical challenge to modern financial ecosystems, characterized by extreme class imbalance where fraudulent transactions typically constitute less than 0.2% of total transaction volume. Standard classification algorithms optimized for overall accuracy fail egregiously in this setting by defaulting to the majority legitimate class. 
> 
> This project implements an end-to-end, reproducible machine learning pipeline evaluated on the European cardholder benchmark dataset (284,807 transactions; 492 fraud cases). To mirror genuine fraud detection environments and eliminate temporal data leakage, transactions are strictly ordered chronologically and divided into a 60:20:20 train, validation, and test partition. 
> 
> Preprocessing estimators (StandardScaler) are fitted exclusively on training distributions. We examine cost-sensitive and ensemble techniques to resolve class disparity: Cost-Sensitive Logistic Regression, Balanced Random Forests, and Gradient Boosted Decision Trees (XGBoost with negative-to-positive loss scaling). Model selection is governed by Average Precision (PR-AUC) under TimeSeries cross-validation rather than standard ROC-AUC or Accuracy. Decision boundaries are refined through post-hoc threshold optimization on validation partitions prior to a single-pass evaluation on untouched test data. 
> 
> The best-performing pipeline (XGBoost) achieves a test PR-AUC of 0.8006 and a Precision of 0.9778, reducing false positives while capturing fraud signals under realistic computational budgets. The finalized pipeline is integrated into an interactive inference frontend built with Streamlit.

---

## 3. Key Technical Highlights for Viva / Presentation

When explaining this project to examiners or your project guide:

1. **Why Not Accuracy?**
   * *Talking Point*: "In our dataset, legitimate transactions make up 99.828% of all data. A dummy model that blindly classifies every transaction as legitimate would achieve 99.83% accuracy while detecting zero fraud. Hence, we use **Average Precision (PR-AUC)** as our primary objective metric."

2. **Why Chronological Split (No Random K-Fold)?**
   * *Talking Point*: "Financial fraud is a non-stationary time-series problem. Shuffling transactions allows the model to train on future fraud patterns to predict past transactions, causing severe data leakage. We enforce strict temporal separation based on elapsed transaction time."

3. **How Imbalance is Handled Without Data Corruption:**
   * *Talking Point*: "Rather than synthetically inflating the whole dataset with SMOTE before splitting (which causes data leakage), we utilize **cost-sensitive algorithmic weighting**:
     * Random Forest: `class_weight='balanced_subsample'`
     * XGBoost: `scale_pos_weight = N_negative / N_positive ≈ 496.76`"

4. **Optimal Decision Threshold vs. Default 0.5:**
   * *Talking Point*: "The default classification cutoff of 0.5 is arbitrary. We run precision-recall curve analysis on the validation set to extract the optimal threshold ($F_1$-score maximization) before applying it to the untouched test set."

---

## 4. BibTeX Citations (For LaTeX Reports)

```bibtex
@inproceedings{dalpozzolo2015calibrating,
  title={Calibrating Probability with Undersampling for Unbalanced Classification},
  author={Dal Pozzolo, Andrea and Caelen, Olivier and Johnson, Reid A and Bontempi, Gianluca},
  booktitle={2015 IEEE Symposium Series on Computational Intelligence},
  pages={159--166},
  year={2015},
  organization={IEEE}
}

@article{dalpozzolo2017credit,
  title={Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy},
  author={Dal Pozzolo, Andrea and Boracchi, Giacomo and Caelen, Olivier and Alippi, Cesare and Bontempi, Gianluca},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={29},
  number={8},
  pages={3784--3797},
  year={2017},
  publisher={IEEE}
}

@book{leborgne2021fraud,
  title={Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook},
  author={Le Borgne, Yann-A{\"e}l and Bontempi, Gianluca},
  year={2021},
  publisher={Universit{\'e} Libre de Bruxelles}
}
```

---

## 5. Summary of Empirical Results from Your Trained Model

* **Trained Pipeline**: `models/best_pipeline.pkl` (StandardScaler + XGBoost)
* **Optimized Threshold**: `0.9997`
* **Test PR-AUC**: **0.8006**
* **Test ROC-AUC**: **0.9791**
* **Test Precision**: **97.78%**
* **Local Inference Script**: `predict.py`
* **Interactive Frontend**: `app.py` (Streamlit)
