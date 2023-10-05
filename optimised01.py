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

# Linear Regression Model
def MLModel(x_array, y_array): 
    x_array = st.add_constant(x_array)
    model = st.OLS(y_array, x_array).fit()
    pred = model.predict(x_array)
    model_details = model.summary()
    print(model_details)
    
MLModel(x, motor_updrs)
MLModel(x, total_updrs)



# Now apply non-linear log-transformation to all variables
column_names = df.columns

for i in range(len(column_names)):
    if i > 0:
        df["adjusted_{}".format(column_names[i])] = df[column_names[i]].apply(np.log) # log transform
        # omit the original column 
        df = df.drop(column_names[i], axis=1)
            
# print(df.info())

# x = df.iloc[:,[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
# Extract the Response variables
print("\n ADJUSTED")
motor_updrs = df.iloc[:,4]
total_updrs = df.iloc[:,5]

# Linear Regression Model
def MLModel(x_array, y_array): 
    x_array = st.add_constant(x_array)
    model = st.OLS(y_array, x_array).fit()
    pred = model.predict(x_array)
    model_details = model.summary()
    print(model_details)
    
MLModel(x, motor_updrs)
MLModel(x, total_updrs)
