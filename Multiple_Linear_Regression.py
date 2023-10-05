import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Create datafrane of po2_data.csv file
df = pd.read_csv("po2_data.csv")

# Extract all columns except for subject#, motor_uprds, and total_uprs for Explanatory variables
x = df.iloc[:,[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
# print(x)
# Extract the Response variables
motor_updrs = df.iloc[:,4].values
total_updrs = df.iloc[:,5].values

def multipleLinearRegression(x_array, y_array, t_size, current_y):
    print("Multiple Linear Regression Model To Predict")
    print("Resonponse variable: ", current_y)
    print("Explanatory variable: Multiple Explanatory Variables") 
    x_train, x_test, y_train, y_test = train_test_split(x_array, 
                                                        y_array, 
                                                        test_size=t_size, 
                                                        random_state=0)

    # Build a linear regression model
    model = LinearRegression()
    # use the Fit method to get inercept and coefficient
    model.fit(x_train, y_train)
    # Print the intercept and coefficient learned by the linear regression model
    print("Intercept: ", model.intercept_)
    print("Coefficient: ", model.coef_)
    # y prediction 
    y_pred = model.predict(x_test)
    # Prediction values vs Actual values
    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print(df_pred)

    # Report on MAE, MSE, RMSE, NRMSE, R^2
    # Mean Absolute Error (MSE)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    # Mean Squared Error (MSE)
    mse = metrics.mean_squared_error(y_test, y_pred)
    # Root Mean Square Error (RMSE)
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # Normalised Root Mean Square Error (NRMSE)
    y_max = y_test.max()
    y_min = y_test.min()
    nrmse = rmse / (y_max - y_min)
    # R Squared
    r2 = metrics.r2_score(y_test, y_pred)
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("NRMSE: ", nrmse)
    print("R^2: ", r2)

    # Mean of y
    print("\n Baseline")
    y_base = np.mean(y_train)
    # Replicate the mean values as many times as there are values in the test set
    y_pred_base = [y_base] * len(y_test)
    # Actual values vs Predicted values of 7 baseline 
    df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
    print(df_base_pred)
    # Baseline models
    # Mean Absolute Error (MAE)
    mae_base = metrics.mean_absolute_error(y_test, y_pred_base)
    # Mean Squared Error (MSE)
    mse_base = metrics.mean_squared_error(y_test, y_pred_base)
    # Root Mean Square Error (RMSE)
    rmse_base =  math.sqrt(metrics.mean_squared_error(y_test, y_pred_base))
    # Normalised Root Mean Square Error (NRMSE)
    y_max = y_test.max()
    y_min = y_test.min()
    nrmse = rmse_base / (y_max - y_min)
    # R Squared
    r2 = metrics.r2_score(y_test, y_pred_base)
    print("MAE (Baseline): ", mae_base)
    print("MSE (Baseline): ", mse_base)
    print("RMSE (Baseline): ", rmse_base)
    print("NMRSE (Baseline): ", nrmse)
    print("R^2 (Baseline): ", r2)
    
    print("\n\n")

###########################################################
###########  60% Training set 40% Testing set  ############
###########################################################
print("###########################################################################")
print("###################  60% Training set 40% Testing set  ####################")
print("###########################################################################")
multipleLinearRegression(x, motor_updrs, 0.4, "Motor_UPDRS")
multipleLinearRegression(x, total_updrs, 0.4, "Total_updrs")

###########################################################
###########  50% Training set 50% Testing set  ############
###########################################################
print("###########################################################################")
print("###################  50% Training set 50% Testing set  ####################")
print("###########################################################################")
multipleLinearRegression(x, motor_updrs, 0.5, "Motor_UPDRS")
multipleLinearRegression(x, total_updrs, 0.5, "Total_updrs")

###########################################################
###########  70% Training set 30% Testing set  ############
###########################################################
print("###########################################################################")
print("###################  70% Training set 30% Testing set  ####################")
print("###########################################################################")
multipleLinearRegression(x, motor_updrs, 0.3, "Motor_UPDRS")
multipleLinearRegression(x, total_updrs, 0.3, "Total_updrs")

###########################################################
###########  80% Training set 20% Testing set  ############
###########################################################
print("###########################################################################")
print("###################  80% Training set 20% Testing set  ####################")
print("###########################################################################")
multipleLinearRegression(x, motor_updrs, 0.2, "Motor_UPDRS")
multipleLinearRegression(x, total_updrs, 0.2, "Total_updrs")