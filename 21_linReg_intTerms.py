# SOURCE: https://developer.nvidia.com/blog/a-comprehensive-guide-to-interaction-terms-in-linear-regression/#entry-content-comments

import numpy as np
import pandas as pd
 
import statsmodels.api as sm
import statsmodels.formula.api as smf
 
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
 
# settings
plt.style.use("seaborn-v0_8")
sns.set_palette("colorblind")
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams['figure.dpi'] = 300

# Laden der Daten
df = pd.read_csv('filtered_dataset.csv')

# Aufteilen der Daten in Features (X) und Zielvariable (Y)
X = df.drop(['Y', 'funcCalls', 'assignm'], axis=1)
y = df['Y']

# Intervall der X-Achse für schöne Darstellung
filtered_df = df[df['yLeafs'] <= 100]
# filtered_df = filtered_df[filtered_df['leafsInLists'] <= 1500]

# 2d scatter plot
# sns.lmplot(x="leafsInLists", y="Y", hue="spUsed", data=filtered_df, fit_reg=False)
# plt.ylabel("runtime [µs]")
# plt.xlabel("Leafs in Lists");

# # HuePlot abspeichern und anzeigen
# plt.savefig("22_spUsed_interaction.png")
# plt.show()

model_1 = smf.ols(formula="Y ~ yLeafs + spUsed", data=filtered_df).fit()
print(model_1.summary())

# Access the coefficients
coefficients = model_1.params

# Print the coefficients
print(coefficients)

X = np.linspace(1, 100, num=20)
sns.lmplot(x="yLeafs", y="Y", hue="spUsed", data=filtered_df, fit_reg=False)
plt.title("Best fit lines for from the model without interactions")
plt.ylabel("runtime [µs]")
plt.xlabel("Vehicle Weight")
plt.plot(X, (coefficients["Intercept"] + coefficients["spUsed"]) + coefficients["yLeafs"] * X, "blue")
plt.show()

model_2 = smf.ols(formula="Y ~ yLeafs + spUsed + yLeafs:spUsed", data=filtered_df).fit()
print(model_2.summary())
# Access the coefficients
coefficients = model_2.params

X = np.linspace(1, 100, num=20)
sns.lmplot(x="yLeafs", y="Y", hue="spUsed", data=filtered_df, fit_reg=False)
plt.title("Best fit lines for from the model with interactions")
plt.ylabel("runtime [µs]")
plt.xlabel("Vehicle Weight")
plt.plot(X, coefficients["Intercept"] + coefficients["yLeafs"] * X, "blue")
plt.plot(X, (coefficients["Intercept"] + coefficients["spUsed"]) + (coefficients["yLeafs"] + coefficients["yLeafs:spUsed"]) * X, "orange");
plt.show()

