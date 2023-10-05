# COLLINEARITY


import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("po2_data_optimised01.csv")

corr = df.corr()

# Creeate the heatmap 
ax = sns.heatmap(
    corr, 
    linewidths=2,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    annot_kws={"size": 7},
    square=False,
    annot=True
)

sns.color_palette("rocket", as_cmap=True)

# set the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=35, # adjust it by 35 degrees
    horizontalalignment='right'
)

ax.set_title('Collinearity Analysis')

plt.show()

# build Linear Regression Model

# Extract all columns except for subject#, motor_uprds, and total_uprs for Explanatory variables
x = df.iloc[:,[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
# Extract the Response variables
motor_updrs = df.iloc[:,4]
total_updrs = df.iloc[:,5]

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(motor_updrs,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)

model = sm.OLS(total_updrs,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)

# # Using statsmodels, build the same model but remove all the strongly correlated variables

# Drop one or more of the correlated variables. Keep only one.
df = df.drop(["jitter(abs)"], axis=1)
df = df.drop(["jitter(%)"], axis=1)
df = df.drop(["shimmer(%)"], axis=1)
df = df.drop(["shimmer(abs)"], axis=1)


print(df.info())

# Extract all columns except for subject#, motor_uprds, and total_uprs for Explanatory variables but remove the last 4 due to 4 varibales being removed previously
x = df.iloc[:,[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17]]


# # Build and evaluate the linear regression model again after some Collinearity are removed
x = sm.add_constant(x)
model = sm.OLS(motor_updrs,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)

model = sm.OLS(total_updrs,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)

# r-squared and r-adjusted values seems to decrease