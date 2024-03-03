import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('dataset_v1.csv')
X = df.drop('Y', axis=1)
y = df['Y']

# Spalten killen
X = X.drop(columns=["assignm", "funcCalls"])

# Aufteilen in Trainings- und Validierungssets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisieren des XGBRegressors
model = XGBRegressor(
    max_depth=5, 
    learning_rate=0.037, 
    n_estimators=130, 
    random_state=42,
    min_child_weight=1
)

# Durchführung R2 CrossValidation
scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
scores_r3 = [round(score, 3) for score in scores]

print("R2-Scores:                   ", scores_r3)
print("Durchschnittlicher R2-Score: ", round(scores.mean(), 3))
print("Standardabweichung:          ", round(scores.std(), 3))
