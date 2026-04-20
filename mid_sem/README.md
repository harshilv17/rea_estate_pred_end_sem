# Intelligent Property Price Prediction

Machine Learning based real estate price prediction system using Bengaluru housing data.

This project was developed as part of the **GenAI Capstone – Milestone 1** and focuses on building a predictive model that estimates property prices based on housing features.

---

# Project Overview

Real estate price estimation is an important problem for property buyers, investors, and urban planners. Traditional valuation methods rely on manual comparisons and expert opinions, which can be subjective and time-consuming.

This project applies **machine learning techniques** to analyze housing data and predict property prices based on features such as:

- Location
- Property size (square footage)
- Number of bedrooms
- Bathrooms
- Balconies
- Area type

Two regression models were implemented:

- Linear Regression
- Random Forest Regressor

The trained model is deployed using **Streamlit** to allow users to estimate property prices interactively.

---

# Dataset

The dataset contains housing listings from **Bengaluru, India**.

Key features include:

- Area Type
- Location
- Total Square Footage
- Number of Bedrooms (BHK)
- Number of Bathrooms
- Number of Balconies
- Property Price (Target Variable)

The dataset required preprocessing due to missing values, inconsistent square footage formats, and high-cardinality categorical features.

---

# Data Preprocessing

Several preprocessing steps were applied:

- Handling missing values
- Extracting BHK from textual size data
- Converting square footage ranges into numeric values
- Grouping rare locations into an "Other" category
- Removing unrealistic outliers

Outlier detection was performed using **price-per-square-foot analysis** and logical constraints.

---

# Machine Learning Models

## Linear Regression

Linear Regression models the relationship between features and the target variable using a linear equation:

\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n
\]

Where:

- \(y\) = predicted price
- \(x_i\) = input features
- \(\beta_i\) = coefficients

---

## Random Forest Regressor

Random Forest is an ensemble learning algorithm that builds multiple decision trees and averages their predictions.

\[
\hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(x)
\]

Where:

- \(T\) = number of trees
- \(f_t(x)\) = prediction of each tree

Random Forest can capture **nonlinear relationships** between features.

---

# Model Evaluation

The models were evaluated using regression metrics:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual values.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

---

### Root Mean Squared Error (RMSE)

RMSE penalizes larger prediction errors more heavily.

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
\]

---

### R² Score

R² measures how well the model explains variance in the dataset.

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

---

# Model Performance

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Linear Regression | 15.96 | 30.00 | 0.83 |
| Random Forest | 14.51 | 30.77 | 0.82 |

The Linear Regression model explains approximately **83% of the variance in property prices**, indicating strong predictive performance.

---

# Web Application

A **Streamlit-based web application** was developed for real-time property price prediction.

Users can input property details such as:

- Area Type
- Location
- Square Footage
- Number of Bedrooms
- Bathrooms
- Balconies

The model then predicts the estimated property price.

---

# Running the Project

### Install dependencies

```bash
pip install -r requirements.txt