import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("po2_data.csv")

motor_updrs = df.iloc[:,4].values
total_updrs = df.iloc[:,5].values

column_names = df.columns
# print(df[column_names[1]])
# print(df[column_names[1]].iloc[0]) # column 2 row 1

def SimpleLinearRegression(y_array,col_num,current_x,current_y, t_size, ):
    print("\n\n")
    print("##########################################")
    print("Linear Regression Model To Predict")
    print("Resonponse variable: ", current_y)
    print("Explanatory variable: ", current_x)
    train_size = 1 - t_size
    print("Train size: ", train_size * 100, " and Test size of: ", t_size * 100)
    # temporary variable
    x_array = df.iloc[:,col_num].values

    # 60% training set 40% testing set
    x_train, x_test, y_train, y_test = train_test_split(x_array, # x variable
                                                        y_array, # y vairable
                                                        test_size= t_size, # % training % testing
                                                        random_state=0) 
    # change 1d array to 2d array
    x_train= x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    # build linear regression
    model = LinearRegression()
    model.fit(x_train, y_train)
    # get the linear regression equation
    print("Intercept (b0): ", model.intercept_)
    print("Coefficient of x (b1): ", model.coef_)
    # use linear regression to make prediction
    y_pred = model.predict(x_test)

    # compare actual vs predicted y values
    df_compare = pd.DataFrame({"Actual: ": y_test, "Predicted: ": y_pred})
    print(df_compare)

    # Compute Evaluation Metrics
    # Mean Squared Error (MAE)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    # MSE
    mse = metrics.mean_squared_error(y_test, y_pred)
    # RMSE 
    rmse = math.sqrt(mse)
    # NRMSE
    y_max = y_test.max()
    y_min = y_test.min()
    nrmse = rmse / (y_max - y_min) 
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("NRMSE: ", nrmse) # we want nrmse to be as close to zero as prossible

    # Baseline Model
    print("\n Baseline Models")
    y_base = np.mean(y_train) # mean of y
    print("Mean of Y in the training set: ", y_base)

    y_pred_base = [y_base] * len(y_test)
    df_compare_base = pd.DataFrame({"Actual: ": y_test, "Predicted: ": y_pred_base})
    print(df_compare_base)

    # Compute Evaluation Metrics
    # MAE
    mae_base = metrics.mean_absolute_error(y_test, y_pred_base)
    # MSE
    mse_base = metrics.mean_squared_error(y_test, y_pred_base)
    # RMSE 
    rmse_base = math.sqrt(mse_base)
    # NRMSE
    y_max_base = y_test.max()
    y_min_base = y_test.min()
    nrmse_base = rmse_base / (y_max_base - y_min_base) 

    print("MAE (baseline): ", mae_base)
    print("MSE (baseline): ", mse_base)
    print("RMSE (baseline): ", rmse_base)
    print("NRMSE (baseline): ", nrmse_base) # we want nrmse to be as close to zero as prossible

###########################################################
###########  60% Training set 40% Testing set  ############
###########################################################
print("###########################################################################")
print("###################  60% Training set 40% Testing set  ####################")
print("###########################################################################")
print("#############################             MOTOR UPRS             ##############################")
for i in range(len(column_names)):
    # print(i)
    if i > 0:
        SimpleLinearRegression(motor_updrs, i, column_names[i],"Motor_UPDRS", 0.4)
print("#############################             TOTAL UPRS             ##############################")
for i in range(len(column_names)):
    # print(i)
    if i > 0:
        SimpleLinearRegression(total_updrs, i, column_names[i],"total_UPDRS", 0.4)
        
        
###########################################################
###########  50% Training set 50% Testing set  ############
###########################################################
print("###########################################################################")
print("###################  50% Training set 50% Testing set  ####################")
print("###########################################################################")
print("#############################             MOTOR UPRS             ##############################")
for i in range(len(column_names)):
    # print(i)
    if i > 0:
        SimpleLinearRegression(motor_updrs, i, column_names[i],"Motor_UPDRS", 0.5)
print("#############################             TOTAL UPRS             ##############################")
for i in range(len(column_names)):
    # print(i)
    if i > 0:
        SimpleLinearRegression(total_updrs, i, column_names[i],"total_UPDRS", 0.5)


###########################################################
###########  70% Training set 30% Testing set  ############
###########################################################
print("###########################################################################")
print("###################  70% Training set 30% Testing set  ####################")
print("###########################################################################")
print("#############################             MOTOR UPRS             ##############################")
for i in range(len(column_names)):
    # print(i)
    if i > 0:
        SimpleLinearRegression(motor_updrs, i, column_names[i],"Motor_UPDRS", 0.3)
print("#############################             TOTAL UPRS             ##############################")
for i in range(len(column_names)):
    # print(i)
    if i > 0:
        SimpleLinearRegression(total_updrs, i, column_names[i],"total_UPDRS", 0.3)


###########################################################
###########  80% Training set 20% Testing set  ############
###########################################################
print("###########################################################################")
print("###################  80% Training set 20% Testing set  ####################")
print("###########################################################################")
print("#############################             MOTOR UPRS             ##############################")
for i in range(len(column_names)):
    # print(i)
    if i > 0:
        SimpleLinearRegression(motor_updrs, i, column_names[i],"Motor_UPDRS", 0.2)
print("#############################             TOTAL UPRS             ##############################")
for i in range(len(column_names)):
    # print(i)
    if i > 0:
        SimpleLinearRegression(total_updrs, i, column_names[i],"total_UPDRS", 0.2)