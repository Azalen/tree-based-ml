import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Laden des Datensatzes
file_path = 'dataset_v1.csv'
df = pd.read_csv(file_path)
target_variable = 'Y'

# Preprocessing
X = df.drop(target_variable, axis=1)
X = X.drop(columns=["assignm", "funcCalls"])
y = df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definition des Rasters für die Gittersuche
param_grid = {
    'n_estimators': [100, 130, 150],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialisieren des XGBRegressor und GridSearchCV
xgb_reg = XGBRegressor(random_state=42)
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=3, scoring='r2', verbose=2)

# Durchführung der Gittersuche
grid_search.fit(X_train, y_train)

# Ausgabe der besten gefundenen Parameter und des besten R2-Werts
print("Beste gefundene Parameter: ", grid_search.best_params_)
print("Bester gefundener R2-Wert: ", grid_search.best_score_)

# Testen des besten Modells
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_r2_score = r2_score(y_test, y_pred)
print("Test R2-Wert: ", test_r2_score)