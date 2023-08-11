import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def run(content):
    data = content.copy()

    X = data.iloc[:, 5:11]
    Y = data.iloc[:, 11]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    best_params_rf = rf.feature_importances_

    xgb_model = xgb.XGBRegressor(max_depth=int(best_params_rf[0] * 5) + 3,
                                 learning_rate=10 ** best_params_rf[1],
                                 n_estimators=[50, 100, 200][int(best_params_rf[2] * 2)])
    xgb_model.fit(X_train, y_train)
    y_true = data['nugget_integrity'].values
    y_pred_xgb = xgb_model.predict(X_test)

    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    y_pred = xgb_model.predict(X)
    data['RABoost回归值'] = xgb_model.predict(X)

    data_true = data[data['RABoost回归值'] <= 7]
    mse_xgb_true = mean_squared_error(data_true['nugget_integrity'], data_true['RABoost回归值'])
    mae_xgb_true = mean_absolute_error(data_true['nugget_integrity'], data_true['RABoost回归值'])
    r2_xgb_true = r2_score(data_true['nugget_integrity'], data_true['RABoost回归值'])

    result = {
        "r2_xgb": r2_xgb,
        "mae_xgb_true": mae_xgb_true,
        "mse_xgb_true": mse_xgb_true,
        "y_true": y_pred,
        "y_pred": y_true,
        "data_true": data_true,
        "model": xgb_model
    }

    plt.scatter(y_test, y_pred_xgb, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('XGBoost Predictions vs True Values')
    # plt.show()

    return result

if __name__ == "__main__":
    content = pd.read_excel("your_data.xlsx")
    result = run(content)
    print(result)
