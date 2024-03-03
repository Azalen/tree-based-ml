import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.metrics import r2_score


df = pd.read_csv('dataset_v1.csv')
X = df.drop('Y', axis=1)
X = X.drop(columns=["assignm", "funcCalls"])
y = df['Y']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definieren des Hyperparameterraums
space = {
    'max_depth': hp.choice('max_depth', range(3, 11)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', range(100, 300)),
    'min_child_weight': hp.choice('min_child_weight', range(1, 8)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'gamma': hp.uniform('gamma', 0, 5),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
}

# Definieren der Zielfunktion
def objective(params):
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = r2_score(y_val, y_pred)
    return {'loss': -score, 'status': STATUS_OK}

# Durchführen der Optimierung
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

print("Beste Parameter:", best_params)
