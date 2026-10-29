# House Price Regression — Supervised Learning

A supervised-learning project that estimates house sale prices using the Ames Housing dataset. The notebook covers problem framing, EDA, preprocessing, feature engineering, five regression models, cross-validation, RandomizedSearchCV, model diagnostics and a saved scikit-learn pipeline.

## Dataset
Download `train.csv` from the Kaggle House Prices competition:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Place `train.csv` in the same folder as `HousePrice_SupervisedLearning.ipynb`. The project uses only `train.csv` and creates its own 80/20 split with `random_state=42`. This is the Ames benchmark dataset (USD-scale SalePrice), not Indian real-estate data.

## Files
- `HousePrice_SupervisedLearning.ipynb` — project notebook, organized by the eight assignment steps.
- `house_price_model.pkl` — fitted Ridge pipeline wrapped to accept raw Kaggle predictor rows.
- `house_price_features.py` — creates the engineered features and ordinal quality mappings expected by the model; keep it beside the PKL file.
- `summary_report.md` — project summary, preprocessing, evaluation and limitations.
- `requirements.txt` — Python package dependencies.

## Run locally
1. Install Python 3.10 or later.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download `train.csv` from Kaggle and place it beside the notebook.
5. Open Jupyter and run all cells from top to bottom:
   ```bash
   jupyter notebook HousePrice_SupervisedLearning.ipynb
   ```
6. The last step saves `house_price_model.pkl` in the current working directory and checks five predictions against actual prices. Keep `house_price_features.py` in the same folder as the PKL file when loading it.

## Example: load the saved model
```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load("house_price_model.pkl")
# Raw Kaggle rows are accepted; Id and SalePrice are dropped automatically.
predicted_prices = model.predict_prices(new_houses)
print(predicted_prices)
```

## Recorded evaluation
In the recorded split, Ridge was selected by 5-fold CV RMSE (about $21,803). Its held-out test RMSE was about $19,480, MAE about $14,019 and R² about 0.9313. Lasso had a lower test RMSE on that split, which is why the notebook selects the final model using cross-validation rather than relying only on the test set. Results may vary if the data or code is changed.

## Recording
Add the URL to your own 5–10 minute screen-and-face project walkthrough here before submission: **[Add recording link]**.

## Important limitation
This is an educational benchmark. Do not use the fitted model to estimate actual Indian property prices without training and validating it on representative local market data.
