This is a high-quality README tailored for a GitHub repository. It emphasizes the **Production-Ready** nature of your code, your **Top 5% ranking**, and the technical depth of the project.

---

# 📈 Winton Stock Market Challenge: High-Frequency Prediction

## 🎯 Project Overview

This repository contains a high-performance machine learning pipeline designed for the **Winton Stock Market Challenge**. The goal is to predict the intra-day and daily returns of multiple stocks based on historical returns and anonymized financial features.

**Performance Achievement:**

* **Validation MAE:** `0.001104`
* **Ranking:** Top 5% Solution (Expected Kaggle LB ~1750-1760)

## 🏗️ Technical Workflow

### 1. Data Preprocessing & Imputation

Financial data is notorious for missing values. This pipeline utilizes `IterativeImputer` (MICE) to estimate missing values through cross-feature modeling, ensuring a more robust dataset than simple mean/median filling.

### 2. Feature Engineering

* **Rolling Momentum:** Captured short-term price trends using a 10-period rolling window.
* **Volatility Analysis:** Calculated 20-period standard deviations to assess market risk.
* **Encoding:** Applied `LabelEncoder` to anonymized categorical features to capture hidden structural relationships.

### 3. Dimensionality Reduction (PCA)

To combat the "Curse of Dimensionality" and reduce noise in high-frequency trading data, **Principal Component Analysis (PCA)** was used to reduce the feature set to the top 50 components, capturing over 52% of total variance.

### 4. Model Architecture: Multi-Output Regression

Since the goal is to predict a sequence of returns (Ret_121 to Ret_180 + PlusOne & PlusTwo), I implemented a **MultiOutputRegressor** wrapper around a **Random Forest** base learner. This allows the model to predict all 62 target variables simultaneously while maintaining computational efficiency.

## 📊 Key Results

The model automatically ranks features based on their information gain. Below is the importance of the top PCA components:

## 🛠️ Tech Stack

* **Language:** Python
* **Core Libraries:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Feature Extraction:** PCA, Iterative Imputer, MultiOutput Regressor

## 🚀 How to Run

1. **Clone the repo:**
```bash
git clone https://github.com/Balaji-2001/Winton-Stock-Challenge.git

```


2. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```


3. **Run the Notebook:**
Open `Winton Stock.ipynb` and execute the cells in order.

## 📬 Contact

**Balaji V** *AI/ML Engineer | Chennai, India* 📧 [balajivasanthakumar2001@gmail.com](mailto:balajivasanthakumar2001@gmail.com)

🔗 [LinkedIn](https://www.linkedin.com/in/balaji070701)
