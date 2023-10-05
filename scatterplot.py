import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("po2_data.csv")

motor_updrs = df.iloc[:,4]
total_updrs = df.iloc[:,5]

column_names = df.columns

# plot scatterplot variables X vs Y

for i in range(len(column_names)):
    if i > 0:   
        x_array = df.iloc[:,i]
        plt.scatter(x_array, motor_updrs)
        plt.xlabel(column_names[i])
        plt.ylabel("Motor_UPDRS")
        plt.show()
        
for i in range(len(column_names)):
    if i > 0:   
        x_array = df.iloc[:,i]
        plt.scatter(x_array, total_updrs)
        plt.xlabel(column_names[i])
        plt.ylabel("Total_UPDRS")
        plt.show()