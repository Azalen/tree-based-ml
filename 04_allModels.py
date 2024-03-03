import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = 'dataset_v1.csv'
target_variable = 'Y'
df = pd.read_csv(file_path)

X = df.drop(target_variable, axis=1)
y = df[target_variable]

# Ausschluss der Spalten "assignments" und "function calls"
X = X.drop(columns=["assignm", "funcCalls"])
print(X.head(3), "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Auswahl Modelle
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost_HPTuned": XGBRegressor(max_depth=5, learning_rate=0.035, n_estimators=130, random_state=42)
}

# Training, Vorhersage & feature importances
results = {}
feature_importances = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"MSE [µs²]": mse, "R2": r2, "MAE [µs]": mae}

    # Berechne feature importances falls tree based
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances[name] = dict(zip(X.columns, importances))

# Ergebnisse in Tabelle anzeigen
results_df = pd.DataFrame(results).T
print(results_df)

# feature importances in Tabelle anzeigen
feature_importances_df_transposed = pd.DataFrame(feature_importances).T
print(feature_importances_df_transposed)
