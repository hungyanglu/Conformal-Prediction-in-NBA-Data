# Conformal-Prediction-in-NBA-Data

## Project Overview
This project implements distribution-free **Conformal Prediction (CP)** in regression to construct mathematically guaranteed prediction intervals ($1-\alpha \in \{0.90, 0.95, 0.99\}$) across high-stakes National Basketball Association (NBA) forecasting problems.

By pairing modern machine learning algorithms with 7 rigorous conformal frameworks, we quantify predictive uncertainty without making parametric distributional assumptions about the underlying data.

---

## Research Questions

1. **Player Efficiency Rating (PER) Forecasting:** What players will perform as the Top 5 PER leaders in the $2021–2022$ season ($X_{2020-2021} \rightarrow Y_{2021-2022}$)?
2. **Team Standings & Championship Prediction:** Which team will achieve the highest winning percentage and secure the championship for the $2021–2022$ season ($X_{2020-2021} \rightarrow Y_{2021-2022}$)?
3. **MVP Voting Share & Contender Forecasting:** Which player will capture the NBA Most Valuable Player (MVP) award based on predicted voting shares ($X_{2020-2021} \rightarrow Y_{2021-2022}$)?

---

## Methodology & Frameworks

### 1. Underlying Base Predictors ($\mathcal{A}$)
- **Multiple Linear Regression (Parametric Baseline):** Ridge/Lasso-regularized linear benchmark.
- **Random Forest Regressor (Non-Parametric Ensemble):** Captures non-linear aging curves and feature interactions.
- **Neural Network / MLP (Deep Learning):** High-dimensional representation learning.
- **ARIMA (Linear Time-Series):** Autoregressive sequential trajectory model.
- **LSTM / Recurrent Network (Deep Sequential):** Explicit multi-year longitudinal temporal dependency modeling.

### 2. Conformal Prediction Frameworks (7 Methods)
- **Standard Split Conformal Prediction** (Lei et al., 2018)
- **Locally Adaptive Split Conformal Prediction** (Lei et al., 2018)
- **Conformalized Quantile Regression - CQR** (Romano et al., 2019)
- **Approximation via Rounding** (Chen et al., 2017)
- **Conformal Prediction with Discretized Data - CPDD** (Chen et al., 2017)
- **Conformal Prediction with Discretized Model - CPDM** (Chen et al., 2017)
- **Adaptive Conformal Inference - ACI** (Gibbs & Candès, 2021)

---

## Data Pipeline

Data collected from [Basketball Reference](https://www.basketball-reference.com/):
- **Player-Level Data (PER & MVP Tasks):** 20 historical NBA seasons ($2000–2001$ to $2020–2021$), $n \approx 2,700$ player-season pairs with 23 advanced metrics.
- **Team-Level Data (Win% Task):** 30 historical NBA seasons ($1991–1992$ to $2020–2021$), $n \approx 900$ team-season records with 51 team performance indicators.
- **Validation Test Set:** $2020–2021$ known season ($X_{2019-2020} \rightarrow Y_{2020-2021}$).
- **Out-of-Sample Target Set:** $2021–2022$ unobserved season ($X_{2020-2021} \rightarrow \hat{Y}_{2021-2022}$).


