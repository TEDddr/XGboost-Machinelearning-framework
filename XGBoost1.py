# XGBoost1.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from future.moves import pickle

def run(content):
    data = content.copy()

    X = data.iloc[:, 5:11]
    Y = data.iloc[:, 11]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    xgb_model = xgb.XGBRegressor()  # Use default hyperparameters

    xgb_model.fit(X_train, y_train)  # Train the model

    y_pred_xgb = xgb_model.predict(X_test)

    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # Convert DataFrame columns to NumPy arrays
    X = X.values
    y_true = data['nugget_integrity'].values
    y_pred = xgb_model.predict(X)

    data['XGBoost回归值'] = y_pred

    data_true = data[data['XGBoost回归值'] <= 7]
    mse_xgb_true = mean_squared_error(data_true['nugget_integrity'], data_true['XGBoost回归值'])
    mae_xgb_true = mean_absolute_error(data_true['nugget_integrity'], data_true['XGBoost回归值'])
    r2_xgb_true = r2_score(data_true['nugget_integrity'], data_true['XGBoost回归值'])

    data_true.to_excel('XGBoost回归值.xlsx', index=False)

    result = {
        "r2_xgb": r2_xgb,
        "mae_xgb_true": mae_xgb_true,
        "mse_xgb_true": mse_xgb_true,
        "y_true": y_true,
        "y_pred": y_pred,
        "data_true": data_true,
        "model": xgb_model
    }

    return result


if __name__ == "__main__":
    content = pd.read_excel("your_data.xlsx")
    run(content)