# Winton Stock Market Forecasting

A machine learning project for multi-output stock price forecasting using the Winton Stock Market Challenge dataset.

## 🚀 Project Overview

This project tackles the [Winton Stock Market Challenge](https://www.kaggle.com/competitions/the-winton-stock-market-challenge) by building a robust multi-output regression model to predict future stock returns. The solution leverages advanced feature engineering and statistical techniques to enhance prediction accuracy and reliability.

## 🎯 Features

- *Multi-output regression:* Predicts multiple future stock returns simultaneously.
- *Advanced preprocessing:* Handles missing values, outliers, and feature scaling.
- *Feature engineering:* Extracts statistical and interaction features from raw financial data.
- *Model evaluation:* Uses Mean Squared Error (MSE) and Mean Absolute Error (MAE) for performance validation.
- *Ready for deployment:* Modular code for easy extension and integration.

## 📂 Dataset

- *Source:* [Winton Stock Market Challenge (Kaggle)](https://www.kaggle.com/competitions/the-winton-stock-market-challenge/data)
- *Description:* Contains stock price features, intraday returns, and target variables for future returns prediction.

## 🛠 Installation

1. *Clone the repository:*
   bash
   git clone https://github.com/Balaji-2001/winton-stock-market-forecasting.git
   cd winton-stock-market-forecasting
   

2. *Install dependencies:*
   bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   

3. *Download the dataset:*
   - Place train.csv and test.csv from Kaggle in the project directory.

## 🏃 Usage

### 1. Train the Model

bash
python winton_model.py

- Trains a MultiOutputRegressor (with RandomForestRegressor) and saves predictions.

### 2. Evaluate & Predict

- The script prints validation MAE and outputs predictions for the test set.
- Modify the code to match the competition’s submission format if required.

## 🧠 Modeling Approach

- *Preprocessing:*  
  - Imputes missing values with median.
  - Clips outliers at 1st and 99th percentiles.
  - Scales features using RobustScaler.
- *Feature Engineering:*  
  - Statistical features (mean, std) from intraday returns.
  - Feature interactions (e.g., product of key features).
- *Model:*  
  - MultiOutputRegressor with RandomForestRegressor as the base estimator.
- *Validation:*  
  - 20% split for validation.
  - Metrics: MAE, MSE.

## 📊 Results

- *Validation MAE:* Printed after training.
- *Test Predictions:* Provided as a DataFrame for easy submission.

## 🖥 Technologies Used

- Python
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

## 🙏 Credits

- Dataset: [Winton Stock Market Challenge (Kaggle)](https://www.kaggle.com/competitions/the-winton-stock-market-challenge)
- Developed by [Balaji V](https://github.com/Balaji-2001)

*Feel free to fork, contribute, or use this project for learning and research!*

Let me know if you’d like to add sample output, code snippets, or further customization for your GitHub!
