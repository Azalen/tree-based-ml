# Importieren der notwendigen Bibliotheken
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Laden der Daten
df = pd.read_csv('dataset_v1.csv')

# Aufteilen der Daten in Features (X) und Zielvariable (Y)
X = df.drop('Y', axis=1)
y = df['Y']

# Aufteilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Erstellen des linearen Regressionsmodells
model = LinearRegression()

# Trainieren des Modells mit den Trainingsdaten
model.fit(X_train, y_train)

# Vorhersagen mit den Testdaten
y_pred = model.predict(X_test)

# Ausgabe der Koeffizienten, gerundet auf zwei Dezimalstellen
coefficients = model.coef_
rounded_coefficients = [round(coef, 2) for coef in coefficients]
print("Koeffizienten:", rounded_coefficients)

# Bewertung des Modells mit verschiedenen Qualitätsmetriken
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Bestimmtheitsmaß (R^2):", round(r2, 2))
print("Mittlerer quadratischer Fehler (MSE):", round(mse, 2))
print("Mittlerer absoluter Fehler (MAE):", round(mae, 2))
