import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp
import joblib


def run(content):
    data = content.copy()

    X = data.iloc[:, 5:11]
    Y = data.iloc[:, 11]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    space = {
        'max_depth': hp.choice('max_depth', range(3, 8)),
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'n_estimators': hp.choice('n_estimators', [50, 100, 200])
    }

    def objective(params):
        model = xgb.XGBRegressor(
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators']
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)

    best_max_depth = range(3, 8)[best['max_depth']]
    best_learning_rate = best['learning_rate']
    best_n_estimators = [50, 100, 200][best['n_estimators']]

    best_model = xgb.XGBRegressor(
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        n_estimators=best_n_estimators
    )

    best_model.fit(X_train, y_train)
    y_true = data['nugget_integrity'].values
    y_pred_xgb = best_model.predict(X_test)

    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    y_pred = best_model.predict(X)
    data['贝叶斯-XGBoost回归值'] = y_pred

    data_true = data[data['贝叶斯-XGBoost回归值'] <= 7]
    mse_xgb_true = mean_squared_error(data_true['nugget_integrity'], data_true['贝叶斯-XGBoost回归值'])
    mae_xgb_true = mean_absolute_error(data_true['nugget_integrity'], data_true['贝叶斯-XGBoost回归值'])
    r2_xgb_true = r2_score(data_true['nugget_integrity'], data_true['贝叶斯-XGBoost回归值'])

    result = {
        "r2_xgb": r2_xgb,
        "mae_xgb_true": mae_xgb_true,
        "mse_xgb_true": mse_xgb_true,
        "y_true": y_true,
        "y_pred": y_pred,
        "data_true": data_true,
        "model": best_model

    }

    return result


if __name__ == "__main__":
    content = pd.read_excel("your_data.xlsx")
    result = run(content)
    print(result)
