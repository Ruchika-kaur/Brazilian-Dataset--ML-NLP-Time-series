# Machine Learning, NLP & Time Series Project

This project demonstrates the application of multiple data science techniques including:

- Customer Segmentation (Clustering)
- Supervised Learning (Classification & Regression)
- Natural Language Processing (Sentiment Analysis)
- Time Series Forecasting

The goal was to compare different models for each problem, evaluate performance using appropriate metrics, and select the best model based on both accuracy and business interpretability.

---

# Project Overview

This project is divided into five main sections:

1. Customer Segmentation (Clustering)
2. Classification Modeling
3. Regression Modeling
4. Sentiment Analysis (NLP)
5. Time Series Forecasting

# Dataset Information

- The project uses the Brazilian E-Commerce Public Dataset by Olist.
- The dataset contains real transactional data from 2016 to 2018.
- It includes information on customers, orders, payments, reviews, products, sellers, and geolocation.
- The dataset consists of approximately 100,000 orders and related records.
- Data is distributed across multiple relational tables that were merged for analysis.
- Customer location data includes city, state, latitude, and longitude information.
- Payment data provides monetary values used for revenue analysis and RFM segmentation.
- Review data contains customer feedback used for sentiment classification (NLP).
- Order timestamps were used for delivery time calculation and time series forecasting.

The dataset contains missing values and skewed numerical features, requiring preprocessing and feature engineering.
---

# Customer Segmentation (Clustering)

### Models Used:
- K-Means (k=4)
- DBSCAN

### Evaluation Metric:
- Silhouette Score

### Results:
- DBSCAN achieved the highest silhouette score (~0.69), indicating strong density-based separation.
- K-Means achieved a slightly lower score (~0.52) but produced four clear customer segments.

### Final Selection:
K-Means (k=4) was selected due to better business interpretability and more actionable customer groups.

---

# Classification Modeling

### Models Compared:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

### Results:
- Random Forest achieved the highest accuracy (~0.83)
- Strong weighted F1-score (~0.81)
- Balanced precision and recall across classes

### Final Selection:
Random Forest was selected as the final classification model due to its superior overall performance and robustness.

---

# Regression Modeling

### Models Compared:
- Linear Regression
- Random Forest Regressor

### Evaluation Metrics:
- MAE
- RMSE
- R² Score

### Results:
- Random Forest Regressor achieved lower MAE and RMSE
- Higher R² score compared to Linear Regression

### Final Selection:
Random Forest Regressor was selected due to better predictive accuracy and ability to capture non-linear patterns.

---

# NLP – Sentiment Analysis

### Task:
Classify customer reviews into:
- Positive
- Neutral
- Negative

### Preprocessing:
- Text cleaning
- Lowercasing
- Stopword removal
- TF-IDF Vectorization (n-grams)

### Models Compared:
- Logistic Regression
- Decision Tree
- Random Forest
- KNN

### Results:
Random Forest achieved the best performance with highest accuracy and balanced F1-score.

---

# Time Series Forecasting

### Task:
Forecast future monthly order volumes.

### Steps:
- Monthly aggregation of order data
- Stationarity testing (ADF Test)
- Train-test split
- Model comparison

### Models Compared:
- ARIMA
- SARIMA
- Prophet

### Evaluation Metrics:
- MAE
- RMSE

### Results:
- ARIMA achieved the lowest RMSE.
- SARIMA was selected due to its ability to model seasonality.
- Final model used for 12-month future forecasting.

---

# Key Skills Demonstrated

- Data Cleaning & Preprocessing
- Feature Engineering
- Model Comparison
- Evaluation Metrics
- Ensemble Learning
- NLP with TF-IDF
- Time Series Forecasting
- Business Interpretation of Models

---

# How to Run

1. Clone the repository
2. Install dependencies:

   pip install -r requirements.txt

3. Open and run the Jupyter notebooks in order.

---

# Final Conclusion

- Ensemble methods (Random Forest) consistently performed best in classification and regression tasks.
- K-Means provided actionable customer segmentation.
- NLP models successfully classified customer sentiment using TF-IDF features.
- SARIMA was selected for forecasting due to seasonality modeling capability.

This project demonstrates end-to-end application of machine learning, NLP, and time series modeling techniques on real-world structured and unstructured data.
