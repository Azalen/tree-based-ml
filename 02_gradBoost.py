import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = 'dataset_v1.csv'
zielvariable = 'Y'
df = pd.read_csv(file_path)

# Auswahl der Merkmale und der Zielvariablen
X = df.drop(zielvariable, axis=1)
y = df[zielvariable]

X = X.drop(columns=["assignm", "funcCalls"])

# Aufteilen des Datensatzes in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialisieren des Gradient Boosting Regressors
gb_reg = GradientBoostingRegressor(
    n_estimators=200, 
    learning_rate=0.02, 
    max_depth=2, 
    random_state=42
)

# Trainieren des Modells mit Trainingsdaten
gb_reg.fit(X_train, y_train)

# Vorhersagen auf Testdaten
y_pred = gb_reg.predict(X_test)

# Leistungsmerkmale
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2:                                     {round(r2, 3)}")
print(f"Mittlerer absoluter Fehler (MAE):       {int(mae)} µs")
