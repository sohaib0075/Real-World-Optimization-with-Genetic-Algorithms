# 🤖 ML Classification Suite

A collection of three supervised learning experiments exploring Gaussian Naïve Bayes and Decision Tree classifiers across real-world datasets — income prediction, car evaluation, and diabetes diagnosis.

---

## 📁 Project Structure

```
ml-classification-suite/
├── q1_gaussian_nb/
│   ├── adult_income_gnb.py       # Gaussian NB on Adult Income dataset
│   └── results/
├── q2_decision_tree_car/
│   ├── car_evaluation_dt.py      # Decision Tree on Car Evaluation dataset
│   └── results/
├── q3_decision_tree_diabetes/
│   ├── diabetes_dt.py            # Decision Tree on Diabetes dataset
│   └── results/
├── data/
│   ├── adult.data
│   ├── adult.test
│   ├── car.data
│   └── diabetes.csv
└── README.md
```

---

## 📊 Question 1 — Gaussian Naïve Bayes on Adult Income Dataset

### Objective
Predict whether an individual earns **>$50K/year** using a Gaussian Naïve Bayes classifier.

### Dataset
- **Source:** [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Size:** ~48,000 instances, 14 features
- **Target:** Binary — `<=50K` or `>50K`

### Methodology
- Dropped rows containing `"?"` missing values
- One-hot encoded all 8 categorical features
- Left 6 continuous features (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) as-is
- Evaluated on a held-out test set + 10-fold cross-validation on training set
- Decision threshold tuned from 0.50 → 0.30 to improve minority-class recall

### Results

| Metric | Value |
|---|---|
| Accuracy | 78.9% |
| ROC-AUC | 0.826 |
| CV Accuracy (mean) | 78.9% |
| CV Accuracy (range) | 0.78 – 0.798 |
| Null Baseline | 75.4% |

**Classification Report (threshold = 0.30):**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| <=50K | 0.81 | 0.94 | 0.87 |
| >50K | 0.64 | 0.31 | 0.42 |

### Key Takeaways
- **+3.5% lift** over the majority-class baseline
- ROC-AUC of 0.826 indicates strong discriminative power
- 10-fold CV variance of ~0.018 → low overfitting, good generalization
- Threshold tuning offers a precision/recall trade-off for the minority class

---

## 🚗 Question 2 — Decision Tree Classification on Car Evaluation Dataset

### Objective
Classify cars into four acceptability categories: `unacc`, `acc`, `good`, `vgood`.

### Dataset
- **Source:** [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/car+evaluation)
- **Size:** 1,728 instances, 6 categorical features
- **Class Distribution:** unacc 70.1%, acc 22.2%, good 4.0%, vgood 3.8%

### Methodology
- One-hot encoded all 6 categorical inputs
- 80/20 stratified train/test split
- Trained two Decision Trees: **Gini index** vs. **Entropy (Information Gain)**
- All other hyperparameters left at default (full-depth, unpruned)

### Results

| Criterion | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---|---|---|---|
| Gini | 97.4% | 0.955 | 0.941 | 0.945 |
| **Entropy** | **99.0%** | **0.970** | **0.972** | **0.972** |

**Confusion Matrix (Entropy, 346 test samples):**

|  | Pred: acc | Pred: good | Pred: unacc | Pred: vgood |
|---|---|---|---|---|
| **True: acc** | 71 | 2 | 4 | 0 |
| **True: good** | 0 | 14 | 0 | 0 |
| **True: unacc** | 1 | 0 | 241 | 0 |
| **True: vgood** | 2 | 0 | 0 | 11 |

### Key Takeaways
- Entropy outperforms Gini by ~1.6% accuracy and ~0.03 macro-F1
- Information gain yields purer child nodes on categorical features
- Gini remains a fast, near-equivalent alternative for production use
- Pruning (max_depth, min_samples_leaf) can maintain >95% accuracy with far simpler trees

---

## 🏥 Question 3 — Decision Tree Classification on Diabetes Dataset

### Objective
Predict diabetes diagnosis (`Outcome = 1`) using clinical measurements, with hyperparameter tuning via Grid Search.

### Dataset
- **Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Size:** 768 instances, 8 continuous clinical features
- **Split:** 90% train (691 samples) / 10% test (77 samples), stratified

### Methodology
- Baseline: `DecisionTreeClassifier` with default parameters
- Tuning: 5-fold Grid Search CV over `max_depth`, `min_samples_split`, `min_samples_leaf`
- Best params: `max_depth=5`, `min_samples_split=2`, `min_samples_leaf=4`

### Results

| Model | Accuracy | Precision (class 1) | Recall (class 1) | F1 (class 1) |
|---|---|---|---|---|
| Baseline | 74.0% | 0.640 | 0.593 | 0.615 |
| **Tuned** | **77.9%** | **0.708** | **0.630** | **0.667** |

### Key Takeaways
- Constraining tree depth reduced overfitting and lifted accuracy by **+3.9%**
- Diabetic class F1 improved from 0.615 → 0.667 (+0.052)
- For high-stakes use cases, cost-sensitive learning or class weighting should be explored to minimize false negatives

---

## 🛠️ Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## 🚀 Usage

```bash
# Q1 — Gaussian NB on Adult Income
python q1_gaussian_nb/adult_income_gnb.py

# Q2 — Decision Tree on Car Evaluation
python q2_decision_tree_car/car_evaluation_dt.py

# Q3 — Decision Tree on Diabetes
python q3_decision_tree_diabetes/diabetes_dt.py
```

---

## 📌 Summary Comparison

| Question | Dataset | Algorithm | Best Accuracy | Notable Metric |
|---|---|---|---|---|
| Q1 | Adult Income | Gaussian NB | 78.9% | ROC-AUC: 0.826 |
| Q2 | Car Evaluation | Decision Tree (Entropy) | 99.0% | Macro-F1: 0.972 |
| Q3 | Diabetes | Decision Tree (Tuned) | 77.9% | Diabetic F1: 0.667 |

---

## 📝 License

This project is for academic and educational purposes.
