# Public Data Sources for Machine Learning

A curated list of tabular datasets for learning classical ML, organized by difficulty and task type.

---

## 🟢 Beginner (Start Here)

### Binary Classification

| Dataset | Rows | Features | Target | Link |
|---------|------|----------|--------|------|
| **Titanic** | 891 | 12 | Survival (0/1) | [Kaggle](https://kaggle.com/c/titanic) |
| **Heart Disease** | 303 | 14 | Disease (0/1) | [UCI](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| **Breast Cancer** | 569 | 30 | Malignant/Benign | `sklearn.datasets.load_breast_cancer()` |
| **Bank Marketing** | 45K | 17 | Subscription (0/1) | [UCI](https://archive.ics.uci.edu/ml/datasets/bank+marketing) |

### Multi-class Classification

| Dataset | Rows | Features | Classes | Link |
|---------|------|----------|---------|------|
| **Iris** | 150 | 4 | 3 species | `sklearn.datasets.load_iris()` |
| **Wine** | 178 | 13 | 3 types | `sklearn.datasets.load_wine()` |
| **Digits** | 1797 | 64 | 10 digits | `sklearn.datasets.load_digits()` |

### Regression

| Dataset | Rows | Features | Target | Link |
|---------|------|----------|--------|------|
| **Boston Housing** | 506 | 13 | Price | `sklearn.datasets.load_boston()` |
| **California Housing** | 20K | 8 | Median value | `sklearn.datasets.fetch_california_housing()` |
| **Diabetes** | 442 | 10 | Disease progression | `sklearn.datasets.load_diabetes()` |

---

## 🟡 Intermediate

### Classification (Larger / More Complex)

| Dataset | Rows | Features | Task | Link |
|---------|------|----------|------|------|
| **Adult Income** | 48K | 14 | Salary >50K | [UCI](https://archive.ics.uci.edu/ml/datasets/adult) |
| **Mushroom** | 8K | 22 | Poisonous/Edible | [UCI](https://archive.ics.uci.edu/ml/datasets/mushroom) |
| **Credit Card Fraud** | 284K | 30 | Fraud detection | [Kaggle](https://kaggle.com/mlg-ulb/creditcardfraud) |
| **Telco Churn** | 7K | 21 | Customer churn | [Kaggle](https://kaggle.com/blastchar/telco-customer-churn) |

### Regression (Larger)

| Dataset | Rows | Features | Target | Link |
|---------|------|----------|--------|------|
| **House Prices** | 1.4K | 80 | Sale price | [Kaggle](https://kaggle.com/c/house-prices-advanced-regression-techniques) |
| **Bike Sharing** | 17K | 16 | Rental count | [UCI](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) |
| **Energy Efficiency** | 768 | 8 | Energy load | [UCI](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) |

### Time Series

| Dataset | Rows | Task | Description | Link |
|---------|------|------|-------------|------|
| **Air Quality** | 9K | Forecasting | Pollution levels | [UCI](https://archive.ics.uci.edu/ml/datasets/Air+Quality) |
| **Stock Prices** | Various | Forecasting | Financial data | [Yahoo Finance](https://finance.yahoo.com) |
| **Electricity Load** | 370K | Forecasting | Power consumption | [UCI](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) |

---

## 🔴 Advanced / Competition-Level

### Kaggle Competitions

| Competition | Task | Why It's Good | Link |
|-------------|------|---------------|------|
| **Spaceship Titanic** | Classification | Feature engineering practice | [Kaggle](https://kaggle.com/c/spaceship-titanic) |
| **Tabular Playground** | Various | Monthly competitions | [Kaggle Playground](https://kaggle.com/competitions?hostSegmentIdFilter=8) |
| **IEEE Fraud Detection** | Classification | Large-scale imbalanced | [Kaggle](https://kaggle.com/c/ieee-fraud-detection) |
| **Ames Housing** | Regression | Advanced feature engineering | [Kaggle](https://kaggle.com/c/house-prices-advanced-regression-techniques) |

### Real-World Challenges

| Dataset | Size | Task | Challenge | Link |
|---------|------|------|-----------|------|
| **NYC Taxi** | 1B+ rows | Regression | Big data handling | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| **Santander Customer** | 200K | Classification | Anonymized features | [Kaggle](https://kaggle.com/c/santander-customer-satisfaction) |
| **Porto Seguro** | 595K | Classification | Heavy class imbalance | [Kaggle](https://kaggle.com/c/porto-seguro-safe-driver-prediction) |

---

## Quick Load Examples

```python
# Sklearn Built-in
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
iris = load_iris(as_frame=True)
cancer = load_breast_cancer(as_frame=True)
housing = fetch_california_housing(as_frame=True)

# Pandas from URL
import pandas as pd
titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Kaggle API
# kaggle competitions download -c titanic
# kaggle datasets download -d <dataset-name>

# OpenML
from sklearn.datasets import fetch_openml
adult = fetch_openml(name='adult', version=2, as_frame=True)
```

---

## Recommended Learning Path

1. **Week 1**: Iris, Titanic (basic classification)
2. **Week 2**: California Housing (regression)
3. **Week 3**: Adult Income, Credit Card Fraud (imbalanced data)
4. **Week 4**: House Prices (feature engineering)
5. **Week 5+**: Kaggle competitions
