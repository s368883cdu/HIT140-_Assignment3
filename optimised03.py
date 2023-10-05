# standardization or rescale


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as st

# Create datafrane of po2_data.csv file

df = pd.read_csv("po2_data_optimised01.csv")

# Extract all columns except for subject#, motor_uprds, and total_uprs for Explanatory variables
x = df.iloc[:,[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
# Extract the Response variables
motor_updrs = df.iloc[:,4]
total_updrs = df.iloc[:,5]



x = st.add_constant(x)
model = st.OLS(motor_updrs, x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)
    


# z-score standardisation
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

# drop constant from dataset
x = x.drop(["const"], axis=1)

standard_x = scalar.fit_transform(x.values)
            
standard_x_dataframe = pd.DataFrame(standard_x, index=x.index, columns=x.columns)
# print(df.info())

standard_x_dataframe = st.add_constant(standard_x_dataframe)
model = st.OLS(motor_updrs, standard_x_dataframe).fit()
pred = model.predict(standard_x_dataframe)
model_details = model.summary()
print(model_details)


# results for r-squared and adjusted r-squared seems to remained the same