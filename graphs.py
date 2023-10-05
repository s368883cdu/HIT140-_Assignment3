import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
df = pd.read_csv("po2_data.csv")

# Separate variables
X_cols = df.columns.difference(['subject#', 'motor_updrs', 'total_updrs'])
X = df[X_cols]
y_mu = df["motor_updrs"]  
y_tu = df["total_updrs"] 

# Standardize & Transform
X_std = preprocessing.scale(X)
X_tf = np.zeros_like(X_std)
for i in range(X_std.shape[1]):
    X_tf[:, i], _ = stats.yeojohnson(X_std[:, i])

X_tf_df = pd.DataFrame(X_tf, columns=X_cols)
X_tf_df = X_tf_df.drop(columns=[c for c in X_tf_df.columns if X_tf_df[c].nunique() == 1])

# Split data (60% training and 40% testing)
x_tr, x_te, y_mu_tr, y_mu_te, y_tu_tr, y_tu_te = train_test_split(X_tf_df, y_mu, y_tu, test_size=0.4, random_state=0)

# Linear regression for motor_updrs
x_tr_mu = sm.add_constant(x_tr)
mod_mu = sm.OLS(y_mu_tr, x_tr_mu).fit()

x_te_mu = sm.add_constant(x_te)
y_mu_pred = mod_mu.predict(x_te_mu)

# Linear regression for total_updrs
x_tr_tu = sm.add_constant(x_tr)
mod_tu = sm.OLS(y_tu_tr, x_tr_tu).fit()

x_te_tu = sm.add_constant(x_te)
y_tu_pred = mod_tu.predict(x_te_tu)

#Plot the distribution of UPDRS scores
plt.hist(df['motor_updrs'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Motor UPDRS scores in the dataset')
plt.xlabel('Motor UPDRS Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.hist(df['total_updrs'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Total UPDRS scores in the dataset')
plt.xlabel('Total UPDRS Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#Plot correlation heatmap of the predictor variable
correlations = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='RdBu_r')
fig.colorbar(cax)
ticks = np.arange(0, len(df.columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns, rotation=90)
ax.set_yticklabels(df.columns)
plt.title('Correlation heatmap of the predictor variable', y=1.2)
plt.show()

# Plot the regression models
plt.figure(figsize=(10, 5))
plt.scatter(y_mu_te, y_mu_pred, alpha=0.5, label="Motor UPDRS", color="skyblue")
plt.scatter(y_tu_te, y_tu_pred, alpha=0.5, label="Total UPDRS", color="pink")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Models Comparison")
plt.legend()
plt.grid(True)
plt.show()
