# GAM Model — Findings and Insights

## 1. Overview

This document summarizes the performance and insights from the **plain vanilla Generalized Additive Model (GAM)** used for forecasting weekly sales at the **store–category–week level**.

The GAM models demand as an additive combination of smooth nonlinear functions of key predictors (e.g., lagged sales, rolling averages, and price) along with linear effects for categorical and event-based variables. This enables the model to capture nonlinear relationships while maintaining interpretability.

---

## 2. Model Objective

The objective of the GAM model is to:

- Forecast **weekly sales (`weekly_units`)**
- At the **store × category × week** level
- Using engineered features such as:
  - Lagged demand
  - Rolling statistics
  - Price features
  - Seasonality indicators
  - Store and category identifiers

---

## 3. Evaluation Framework

### 3.1 Granularity

The model is evaluated at three levels:

1. **Overall (pooled)** — all observations combined  
2. **Category level** — FOODS, HOBBIES, HOUSEHOLD  
3. **Store–category level** — e.g., CA_3_FOODS  

---

### 3.2 Metrics Used

#### A. Absolute Metrics (for model comparison)
- **MAE** — Mean Absolute Error  
- **RMSE** — Root Mean Squared Error  
- **MAPE** — Mean Absolute Percentage Error  

#### B. Scale-Adjusted Metrics (for interpretation)
- **sMAPE** — Symmetric MAPE  
- **NRMSE (% of mean sales)**  
- **MAE (% of mean sales)**  

> Because sales volumes differ significantly across store-category combinations, **percentage-based metrics are the primary basis for interpretation**, while raw MAE/RMSE are used for pooled model comparison.

---

## 4. Overall Model Performance

### 4.1 Pooled Test Metrics

| Metric | Value |
|------|------|
| MAE | 595.59 |
| RMSE | 957.19 |
| MAPE | 7.305 |

### 4.2 Scale-Adjusted Metrics

| Metric | Value |
|------|------|
| sMAPE | 7.758 |
| NRMSE (% of mean sales) | 9.533% |
| MAE (% of mean sales) | 5.932% |

### Interpretation

- The model achieves **~7–8% relative error**, which is strong for weekly retail forecasting  
- Performance is **stable across most series**  
- Compared to other models:
  - **Lower accuracy than pooled RF / LightGBM**
  - **Higher interpretability**

---

## 5. Category-Level Insights

### FOODS
- MAPE ≈ 5.2%  
- Strong relative performance  
- Captures overall trend well  
- Slight **underprediction of peaks**

### HOUSEHOLD
- MAPE ≈ 5.6%  
- Similar performance to FOODS  
- Stable and predictable demand patterns  

### HOBBIES
- MAPE ≈ 11.1%  
- Weakest category  
- Consistent **underprediction**  
- Likely due to:
  - Lower volume
  - Higher volatility
  - Less smooth demand structure  

### Summary

| Category | Performance |
|---------|------------|
| FOODS | Strong |
| HOUSEHOLD | Strong |
| HOBBIES | Weak |

---

## 6. Store-Level Insights

### Strong Performers

Examples:
- CA_3_HOUSEHOLD (~1.7% sMAPE)  
- CA_3_FOODS (~2.6%)  
- CA_4_FOODS (~3.2%)  

Characteristics:
- Smooth demand patterns  
- Strong persistence  
- Well captured by GAM  

---

### Weak Performers

Primarily HOBBIES:
- WI_2_HOBBIES (~22%)  
- WI_3_HOBBIES (~18%)  
- CA_2_HOBBIES (~15%)  

Characteristics:
- Noisy demand  
- Irregular patterns  
- Poor fit for smooth additive structure  

---

### Key Insight

Performance varies due to both:
- Category structure  
- Store-specific demand behavior  

---

## 7. Model Behavior

### 7.1 Underprediction Bias

The model tends to:
- Underpredict high-demand weeks  
- Smooth out peaks  

Reason:
- GAM enforces **smooth nonlinear relationships**  
- Includes implicit **mean reversion**

---

### 7.2 Stability vs Flexibility

| Aspect | GAM Behavior |
|------|-------------|
| Stability | High |
| Flexibility | Moderate |
| Spike capture | Weak |

---

## 8. Interpretability Insights

### Top 5 Features

- `price_rolling_4`  
- `rolling_mean_4`  
- `avg_price`  
- `lag_4`  
- `rolling_mean_8`  

---

### 8.1 Demand Persistence

Features:
- rolling_mean_4  
- lag_4  
- rolling_mean_8  

Insight:
> Recent demand strongly drives future demand  

---

### 8.2 Price Effects

Two distinct patterns:

#### Current Price (`avg_price`)
- Negative relationship  
- Higher price → lower demand  

#### Price Regime (`price_rolling_4`)
- Positive relationship  
- Reflects structural demand differences  

---

### 8.3 Mean Reversion

Feature:
- rolling_mean_8  

Insight:
> Extremely high demand levels tend to normalize  

---

## 9. Strengths and Limitations

### Strengths
- Captures nonlinear relationships  
- Highly interpretable  
- Strong performance on FOODS and HOUSEHOLD  
- Stable across most series  

### Limitations
- Weak on HOBBIES  
- Cannot capture sharp spikes  
- Limited interaction modeling  
- Underprediction bias  

---

## 10. Final Assessment

The GAM is a **strong interpretable baseline model**:

| Dimension | Assessment |
|----------|-----------|
| Accuracy | Moderate |
| Interpretability | High |
| Stability | High |
| Flexibility | Moderate |

---

## 11. Conclusion

The GAM shows that demand can be effectively modeled using:

- Recent demand patterns  
- Price behavior  
- Smooth nonlinear relationships  

While it does not outperform tree-based models in accuracy, it provides:

- Strong interpretability  
- Clear insight into demand drivers  
- A reliable and stable baseline model  

---
