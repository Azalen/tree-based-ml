import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file_path = 'dataset_v1.csv'
df = pd.read_csv(file_path)

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")

# Save the figure
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')