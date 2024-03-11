import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file_path = 'dataset_v1.csv'
df = pd.read_csv(file_path)

# corrMatrix berechnen
corr_matrix = df.corr()

# heatmap plotten
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")

# als PNG speichern
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')