import pandas as pd
import numpy as np

file_path = 'dataset_v1.csv'
df = pd.read_csv(file_path)

x = df["assignm"].values
y = df["funcCalls"].values

# Berechnen der Koeffizienten für lin Gleichung y = mx + n
m, n = np.polyfit(x, y, 1)

print(f"Die lineare Funktion lautet: y = {m}x + {n}")
