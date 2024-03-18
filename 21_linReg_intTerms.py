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
df = pd.read_csv('dataset_v1.csv')

# Aufteilen der Daten in Features (X) und Zielvariable (Y)
X = df.drop(['Y', 'funcCalls', 'assignm'], axis=1)
y = df['Y']

# Intervall der X-Achse für schöne Darstellung
filtered_df = df[df['yLeafs'] <= 100]

# 2d scatter plot no hue
sns.lmplot(x="yLeafs", y="Y", data=filtered_df, fit_reg=False, aspect=16/9, height=4)
plt.ylabel("runtime [µs]")
plt.xlabel("yLeafs");
plt.savefig("22_yLeafs-Y_noHue.png", bbox_inches='tight')

# 2d scatter plot spUsed hue
sns.lmplot(x="yLeafs", y="Y", hue="spUsed", data=filtered_df, fit_reg=False, aspect=16/9, height=4)
plt.ylabel("runtime [µs]")
plt.xlabel("yLeafs");
plt.savefig("23_yLeafs-Y_spusedHue.png", bbox_inches='tight')




model_1 = smf.ols(formula="Y ~ yLeafs + spUsed", data=filtered_df).fit()
print(model_1.summary())
coefficients = model_1.params

X = np.linspace(1, 100, num=20)
sns.lmplot(x="yLeafs", y="Y", data=filtered_df, fit_reg=True, ci=None, aspect=16/9, height=4)
plt.ylabel("runtime [µs]")
plt.xlabel("yLeafs")



plt.savefig("24_yLeafs-Y_fitLine.png", bbox_inches='tight')




model_2 = smf.ols(formula="Y ~ yLeafs + spUsed + yLeafs:spUsed", data=filtered_df).fit()
print(model_2.summary())
coefficients = model_2.params

X = np.linspace(1, 100, num=20)
sns.lmplot(x="yLeafs", y="Y", hue="spUsed", data=filtered_df, ci=None, fit_reg=False, aspect=16/9, height=4)
plt.ylabel("runtime [µs]")
plt.xlabel("Vehicle Weight")
plt.plot(X, coefficients["Intercept"] + coefficients["yLeafs"] * X, "blue")
plt.plot(X, (coefficients["Intercept"] + coefficients["spUsed"]) + (coefficients["yLeafs"] + coefficients["yLeafs:spUsed"]) * X, "orange");
plt.savefig("25_yLeafs-Y_spusedHue_fitLine.png", bbox_inches='tight')


# 2d scatter plot no hue
sns.lmplot(x="spUsed", y="Y", data=filtered_df, fit_reg=False, aspect=16/9, height=4)
plt.ylabel("runtime [µs]")
plt.xlabel("spUsed");

# Set the x-axis to show only 0 and 1
plt.xticks([0, 1])
# Specify the total X-range
plt.xlim(-0.5, 1.5)

plt.savefig("26_spUsed-Y.png", bbox_inches='tight')

subset0_df = filtered_df[filtered_df['spUsed'] == 0]
subset1_df = filtered_df[filtered_df['spUsed'] == 1]

Y_mean_sp0 = subset0_df['Y'].mean()
Y_mean_sp1 = subset1_df['Y'].mean()
print(round(Y_mean_sp0,2), round(Y_mean_sp1,2))
plt.hlines(Y_mean_sp0, xmin=-0.25, xmax=0.25, colors='red', linestyles='--')
plt.hlines(Y_mean_sp1, xmin=0.75, xmax=1.25, colors='red', linestyles='--') 

plt.savefig("27_spUsed-Y_withAver.png", bbox_inches='tight')