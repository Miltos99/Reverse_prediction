import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings
from typing import Tuple
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam



print_mse = True
print_R_square = True
# Path to the training data:
# Specify the file path where the training data is stored. 
# This data will be used to train the machine learning model.
# Example: 'C:\\Users\\name\\Desktop\\All_data.xlsx'
data_path_input_2_zero = "C:\\Users\\milto\\Desktop\\All_data_input2_0.xlsx"

# Path to the test data:
# Indicate the file path where the test data is located.
# Test data is used to evaluate the performance of the trained model.
# Example: 'C:\\Users\\name\\Desktop\\output_target_data.xlsx'
test_data_path = 'C:\\Users\\milto\\Desktop\\output_target_data.xlsx' 

# Path for saving predicted data:
# Define the file path where you want to save the predictions made by the model.
# The predictions will be saved in an Excel format for easy access and review.
# Example: 'C:\\Users\\name\\Desktop\\saved_data.xlsx'
save_data_path_zero = "C:\\Users\\milto\\Desktop\\saved_data_input2_0.xlsx"
#The number of input_columns that the data are goint to have
input_column=9
period = 1702
#Set the column that you want to insert the constrains
column = 1
Input_ = 0 #You can set any value
condition = "==" #or "<" , ">"
period=1701
sub_section_start= 100
sub_section_end = 300

def constraints(df: pd.DataFrame,period: int) -> pd.DataFrame:
    
    # If no DataFrame is provided, or if no input information is given, return the original DataFrame
    if df is None or Input_ == None:
        return df

    # If period is not provided, default to the length of the DataFrame
    if period is None:
        period = len(df)

    result_df = pd.DataFrame()  # Initialize an empty DataFrame for the result

    # Apply constraints to each section of the DataFrame based on respective inputs and conditions
    for i in range(0, len(df), period):
        section = df.iloc[i:i + period]
        sub_section = section.iloc[sub_section_start:sub_section_end]

        # Check conditions within the subsection
        include_section = True
        if Input_ is not None and condition is not None and not check_condition(sub_section, column, Input_, condition):
            include_section = False
        # If all conditions in the subsection are met, add the whole section to the result
        if include_section:
            result_df = pd.concat([result_df, section])

    return result_df

def check_condition(sub_section, column_index, input_value, condition):
    """Helper function to check the condition for a specific column"""
    column = sub_section[sub_section.columns[column_index]]

    if condition == '>':
        return not column.le(input_value).any()  # Return True if all values are greater than input_value
    elif condition == '<':
        return not column.ge(input_value).any()  # Return True if all values are less than input_value
    elif condition == '==':
        return column.eq(input_value).all()      # Return True if all values are equal to input_value
    return False

def compute_model_performance( y_test, y_pred) ->Tuple[float,float]:
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def rename_df(df: pd.DataFrame) -> pd.DataFrame:
    for el,index in zip(df.columns[:3], range(input_column+1,input_column+4)):
        df.rename(columns={el:str(index)},inplace=True)
    return df

def data_structure(df: pd.DataFrame,num_fft_features: int = 2) ->pd.DataFrame:
    # Convert index to a continuous cycle of numbers within the period
    cycle_index = df.index % period

    # Calculate the radians conversion factor (2 * pi / period)
    radians_conversion = 2 * np.pi / period
    # Introducing new time series relationships
    df[input_column + 1] = np.sin(cycle_index * radians_conversion)
    df[input_column + 2] = np.cos(cycle_index * radians_conversion)
    # Calculate FFT features for each period and get the most significant frequencies
    for i in range(0, len(df), period):
        period_slice = df.iloc[i:i+period,input_column]  # Assuming the first column is the time-series data
        fft_result = np.fft.fft(period_slice)
        fft_magnitude = np.abs(fft_result)[:num_fft_features]  # Take the magnitude of the first few components
        # Add FFT features to the DataFrame
        for j in range(num_fft_features):
            df.loc[i:i+period, input_column+3+j] = fft_magnitude[j]
    # Fill any NaNs that may have resulted from the loop with zeros or other appropriate value
    df = df.ffill()
    return df

def data_structure_test(df: pd.DataFrame, num_fft_features: int = 2) -> pd.DataFrame:
    # Convert index to a continuous cycle of numbers within the period
    cycle_index = df.index % period
    # Calculate the radians conversion factor (2 * pi / period)
    radians_conversion = 2 * np.pi / period
    # Introducing new time series relationships
    df[1] = np.sin(cycle_index * radians_conversion)
    df[2] = np.cos(cycle_index * radians_conversion)
    df = rename_df(df)
    # Calculate FFT features for each period and get the most significant frequencies
    for i in range(0, len(df), period):
        period_slice = df.iloc[i:i+period,0]  # First column as the time-series data
        fft_result = np.fft.fft(period_slice)
        # Ensure that num_fft_features does not exceed the available number of components
        num_features = min(num_fft_features, len(fft_result) // 2)
        fft_magnitude = np.abs(fft_result)[:num_features]
        # Add FFT features to the DataFrame
        for j in range(num_features):
            df.loc[i:i+period,input_column+3+j] = fft_magnitude[j]
    df = df.ffill()
    return df

def call_data(data_path: str)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load your data
    df = pd.read_excel(data_path, header=None)  # Adjust path and header as necessary`    `
    
    # Assuming your DataFrame df is already loaded
    # Apply the function
    df = constraints(df,period)
    
    df= data_structure(df)
    X = df.iloc[:, input_column:]
    y= df.iloc[:, :input_column] 
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.001, random_state=42)
    return  X_train, X_test, y_train, y_test


def stabilize_all_values(df:pd.DataFrame)->pd.DataFrame:
    df = stabilize_value_0(df)
    df = stabilize_value_1(df)
    df = stabilize_value_2(df)
    df = stabilize_value_3(df)
    df = stabilize_value_4(df)
    df = stabilize_value_5(df)
    df = stabilize_value_6(df)
    df = stabilize_value_7(df)
    df = stabilize_value_8(df)
    return df

def stabilize_value_0(df:pd.DataFrame)->pd.DataFrame:

    """
    Stabilize predictions in the first column of a DataFrame to conform to a specific pattern:
    zero for the first 100 samples, a large value for the next 601 samples, and zero for the rest.

    :param df: The DataFrame with predictions.
    :param period: The period of the pattern (100 for zero, 601 for high_value, rest for zero).
    :return: The DataFrame with the stabilized first column.
    """
    stabilized_df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for i in range(0, len(df), period):
        sub_period_start = 0
        sub_period_end = 0
        for j in range(1,period):
            if df.iloc[j,0] == df.iloc[j+1,0] and j != (period-2):
                sub_period_end += 1
            else:
                sub_period_end+= 1
                mode_series = df.iloc[sub_period_start:sub_period_end, 0].mode()
                if not mode_series.empty and mode_series.iloc[0] >= -0.1 and mode_series.iloc[0] <= 0.1:
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 0] = 0
                else:
                    mode_value = df.iloc[i+sub_period_start:i+sub_period_end, 0].mode().iloc[0] if not df.iloc[i+sub_period_start:i+sub_period_end, 0].mode().empty else 0
                    # Set the next 601 samples to the max value
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 0] = mode_value
                sub_period_start = sub_period_end

    return stabilized_df

def stabilize_value_1(df:pd.DataFrame)->pd.DataFrame:
    stabilized_df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for i in range(0, len(df), period):
        sub_period_start = 0
        sub_period_end = 0 
        for j in range(1,period):
            if df.iloc[j,1] == df.iloc[j+1,1] and j != (period-2):
                sub_period_end += 1
            else:
                sub_period_end+= 1
                mode_series = df.iloc[sub_period_start:sub_period_end, 1].mode()
                if not mode_series.empty and mode_series.iloc[0] >= -0.1 and mode_series.iloc[0] <= 0.1:
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 1] = 0
                else:
                    mode_value = df.iloc[i+sub_period_start:i+sub_period_end, 1].mode().iloc[0] if not df.iloc[i+sub_period_start:i+sub_period_end, 1].mode().empty else 0
                    # Set the next 601 samples to the max value
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 1] = mode_value
                sub_period_start = sub_period_end
                

    return stabilized_df

def stabilize_value_2(df:pd.DataFrame)->pd.DataFrame:
    stabilized_df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for i in range(0, len(df), period):
        sub_period_start = 0
        sub_period_end = 0 
        for j in range(1,period):
            if df.iloc[j,2] == df.iloc[j+1,2] and j != (period-2):
                sub_period_end += 1
            else:
                sub_period_end+= 1
                mode_series = df.iloc[sub_period_start:sub_period_end, 2].mode()
                if not mode_series.empty and mode_series.iloc[0] >= -0.1 and mode_series.iloc[0] <= 0.1:
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 2] = 0
                else:
                    mode_value = df.iloc[i+sub_period_start:i+sub_period_end, 2].mode().iloc[0] if not df.iloc[i+sub_period_start:i+sub_period_end, 2].mode().empty else 0
                    # Set the next 601 samples to the max value
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 2] = mode_value
                sub_period_start = sub_period_end

    return stabilized_df

def stabilize_value_3(df:pd.DataFrame)->pd.DataFrame:
    stabilized_df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for i in range(0, len(df), period):
        sub_period_start = 0
        sub_period_end = 0 
        for j in range(1,period):
            if df.iloc[j,3] == df.iloc[j+1,3] and j != (period-2):
                sub_period_end += 1
            else:
                sub_period_end+= 1
                mode_series = df.iloc[sub_period_start:sub_period_end, 3].mode()
                if not mode_series.empty and mode_series.iloc[0] >= -0.1 and mode_series.iloc[0] <= 0.1 :
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 3] = 0
                else:
                    mode_value = df.iloc[i+sub_period_start:i+sub_period_end, 3].mode().iloc[0] if not df.iloc[i+sub_period_start:i+sub_period_end, 3].mode().empty else 0
                    # Set the next 601 samples to the max value
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 3] = mode_value
                sub_period_start = sub_period_end
               

    return stabilized_df

def stabilize_value_4(df:pd.DataFrame)->pd.DataFrame:
    stabilized_df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for i in range(0, len(df), period):
        sub_period_start = 0
        sub_period_end = 0 
        for j in range(1,period):
            if df.iloc[j,4] ==df.iloc[j+1,4] and j != (period-2):

                sub_period_end += 1
            else: 
                sub_period_end+= 1
                mode_series = df.iloc[sub_period_start:sub_period_end, 4].mode()
                if not mode_series.empty and mode_series.iloc[0] >= -0.1 and mode_series.iloc[0] <= 0.1:
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 4] = 0
                else:
                    mode_value = df.iloc[i+sub_period_start:i+sub_period_end, 4].mode().iloc[0] if not df.iloc[i+sub_period_start:i+sub_period_end, 4].mode().empty else 0
                    # Set the next 601 samples to the max value
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 4] = mode_value
                sub_period_start = sub_period_end
            

    return stabilized_df

def stabilize_value_5(df:pd.DataFrame)->pd.DataFrame:
    stabilized_df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for i in range(0, len(df), period):
        sub_period_start = 0
        sub_period_end = 0 
        for j in range(1,period):
            if df.iloc[j,5] == df.iloc[j+1,5] and j != (period-2):
                sub_period_end += 1
            else:
                sub_period_end+= 1
                mode_series = df.iloc[sub_period_start:sub_period_end, 5].mode()
                if not mode_series.empty and mode_series.iloc[0] >= -0.1 and mode_series.iloc[0] <= 0.1:
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 5] = 0
                else:
                    mode_value = df.iloc[i+sub_period_start:i+sub_period_end, 5].mode().iloc[0] if not df.iloc[i+sub_period_start:i+sub_period_end, 5].mode().empty else 0
                    # Set the next 601 samples to the max value
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 5] = mode_value
                sub_period_start = sub_period_end
               

    return stabilized_df

def stabilize_value_6(df:pd.DataFrame)->pd.DataFrame:
    stabilized_df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for i in range(0, len(df), period):
        sub_period_start = 0
        sub_period_end = 0 
        for j in range(1,period):
            if df.iloc[j,6] == df.iloc[j+1,6] and j != (period-2):
                sub_period_end += 1
            else:
                sub_period_end+= 1
                mode_series = df.iloc[sub_period_start:sub_period_end, 6].mode()
                if not mode_series.empty and mode_series.iloc[0] >= -0.1 and mode_series.iloc[0] <= 0.1:
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 6] = 0
                else:
                    mode_value = df.iloc[i+sub_period_start:i+sub_period_end, 6].mode().iloc[0] if not df.iloc[i+sub_period_start:i+sub_period_end, 6].mode().empty else 0
                    # Set the next 601 samples to the max value
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 6] = mode_value
                sub_period_start = sub_period_end
              

    return stabilized_df

def stabilize_value_7(df:pd.DataFrame )->pd.DataFrame:
    stabilized_df = df.copy()  # Create a copy to avoid modifying the original DataFrame input_df = input_df.copy()
    for i in range(0, len(df), period):
        sub_period_start = 0
        sub_period_end = 0 
        for j in range(1,period):
            if df.iloc[j,7] == df.iloc[j+1,7] and j != (period-2):
                sub_period_end += 1
            else:
                sub_period_end+= 1
                mode_series = df.iloc[sub_period_start:sub_period_end, 7].mode()
                if not mode_series.empty and mode_series.iloc[0] >= -0.1 and mode_series.iloc[0] <= 0.1:
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 7] = 0
                else:
                    mode_value = df.iloc[i+sub_period_start:i+sub_period_end, 7].mode().iloc[0] if not df.iloc[i+sub_period_start:i+sub_period_end, 7].mode().empty else 0
                    # Set the next 601 samples to the max value
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 7] = mode_value
                sub_period_start = sub_period_end
               

    return stabilized_df

def stabilize_value_8(df:pd.DataFrame )->pd.DataFrame:
    stabilized_df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for i in range(0, len(df), period):
        sub_period_start = 0
        sub_period_end = 0 
        for j in range(1,period):
            if df.iloc[j,5] == df.iloc[j+1,8] and j != (period-2):
                sub_period_end += 1
            else:
                sub_period_end+= 1
                mode_series = df.iloc[sub_period_start:sub_period_end,8].mode()
                if not mode_series.empty and mode_series.iloc[0] >= -0.1 and mode_series.iloc[0] <= 0.1:
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 8] = 0
                else:
                    mode_value = df.iloc[i+sub_period_start:i+sub_period_end, 8].mode().iloc[0] if not df.iloc[i+sub_period_start:i+sub_period_end, 8].mode().empty else 0
                    # Set the next 601 samples to the max value
                    stabilized_df.iloc[i+sub_period_start:i+sub_period_end, 8] = mode_value
                sub_period_start = sub_period_end
            

    return stabilized_df


    
def pred_test_data(path: str, trained_model) -> pd.DataFrame:
    test_dta = pd.read_excel(path, header=None)
    test_dta = data_structure_test(test_dta)
    # Convert all column names to strings
    test_dta.columns = test_dta.columns.astype(str)
    predicted = trained_model.predict(test_dta)
    df_pr = pd.DataFrame(predicted)
    df_pr = stabilize_all_values(df_pr)
    # res_plot(df_pr)
    return df_pr

def pred_test_data_lstm(path: str, trained_model) -> pd.DataFrame:
    test_dta = pd.read_excel(path, header=None)
    test_dta = data_structure_test(test_dta)

    # Convert all column names to strings
    test_dta.columns = test_dta.columns.astype(str)
    test_dta = np.reshape(test_dta.values, (test_dta.shape[0], test_dta.shape[1], 1))
    test_dta = stabilize_all_values(df_pr)
    predicted = trained_model.predict(test_dta)
    df_pr = pd.DataFrame(predicted)
    df_pr = stabilize_all_values(df_pr)

    return df_pr

def res_plot(df:pd.DataFrame ) -> None:
    # Number of samples
    num_samples = len(df)

    # Generate a sample index (0 to num_samples-1)
    sample_index = range(num_samples)

    # Plot each column
    for column in df.columns:
        plt.figure(figsize=(10, 4))  # Set the figure size
        plt.plot(sample_index, df[column])  # Plot column values
        plt.title(f'Values in {column}')
        plt.xlabel('Sample Number')
        plt.ylabel(column)
        plt.show()

def save_data_on_excel(predicted_data ,data_save_path:str) -> None:
    try:
        predicted_data.to_excel(data_save_path, index=False)
        print(f"Data saved successfully to {data_save_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

#########################################################################
############################## MODELS ###################################
#########################################################################

def Random_Forest_Regressor(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame)->Tuple[float, float, np.ndarray]:
    # Initialize and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    mse, r2 = compute_model_performance(y_test, predictions)  # Use the helper function for performance printout

    # Overall evaluation
    average_mse = np.mean(mse)
    average_r2 = np.mean(r2)
    predicted_data = pred_test_data(test_data_path, model)
    return average_mse, average_r2,predicted_data

def train_LGBMRegressor(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame)->Tuple[float, float, np.ndarray]:
    mse_scores = []
    r2_scores = []
    models = []  # To store models for each target

    for i in range(y_train.shape[1]):
        # Initialize LightGBM Regressor for each output
        lgb_model = LGBMRegressor(random_state=42)
        # Train the model on each target column
        lgb_model.fit(X_train, y_train.iloc[:, i])
        # Store the model
        models.append(lgb_model)
        # Make predictions
        lgb_predictions = lgb_model.predict(X_test)
        # Evaluate the model
        mse, r2 =  compute_model_performance(y_test.iloc[:, i], lgb_predictions)  # Use the helper function for performance printout
        # Overall evaluation
        mse_scores.append(mse)
        r2_scores.append(r2)
    average_mse = np.mean(mse_scores)
    average_r2 = np.mean(r2_scores)
    lgb_model = pred_test_data(test_data_path, lgb_model)
    return average_mse,average_r2, models

def train_XGBRegressor(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame)->Tuple[float, float, np.ndarray]:
    # Initialize XGBoost Regressor
    xgb_model = XGBRegressor(random_state=42)
    # Train the model
    xgb_model.fit(X_train, y_train)
    # Make predictions
    xgb_predictions = xgb_model.predict(X_test)
    mse, r2 = compute_model_performance(y_test, xgb_predictions)
    xgb_model = pred_test_data(test_data_path, xgb_model)
    return mse, r2,xgb_model

def train_XGBRegressor_with_RandomSearch(X_train:pd.DataFrame, X_test:pd.DataFrame,\
                                        y_train:pd.DataFrame, y_test:pd.DataFrame)->Tuple[float, float, np.ndarray]:
    # Initialize XGBoost Regressor
    xgb_model = XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 300, 500, 700, 1000],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    random_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=50, scoring='neg_mean_squared_error', cv=5, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Best model
    best_xgb_model = random_search.best_estimator_

    # Make predictions using the best model
    xgb_predictions = best_xgb_model.predict(X_test)

    mse, r2 = compute_model_performance(y_test, xgb_predictions)
    best_xgb_model = pred_test_data(test_data_path, best_xgb_model)
    return mse, r2 ,  best_xgb_model

def tune_RandomForest_hyperparameters(X_train:pd.DataFrame,X_test:pd.DataFrame,y_train:pd.DataFrame,y_test:pd.DataFrame) -> RandomForestRegressor:
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create the Random Forest model
    rf_model = RandomForestRegressor(random_state=42)

    # Create the RandomizedSearchCV object
    randomized_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, 
                                           n_iter=10, scoring='neg_mean_squared_error', 
                                           cv=5, random_state=42, n_jobs=-1, verbose=1)

    # Fit the model to the training data
    randomized_search.fit(X_train, y_train)

    # Get the best model from the randomized search
    best_model = randomized_search.best_estimator_

    # Make predictions on the test set using the best model
    predictions = best_model.predict(X_test)

    # Compute model performance
    mse, r2 = compute_model_performance(y_test, predictions)

    # Overall evaluation
    average_mse = np.mean(mse)
    average_r2 = np.mean(r2)

    # Print the best hyperparameters found
    print("Best hyperparameters from RandomizedSearchCV:", randomized_search.best_params_)
    best_model = pred_test_data(test_data_path, best_model)
    return average_mse, average_r2, best_model

def Single_target_prediction(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame) ->Tuple[float, float, np.ndarray]: 
    models = [RandomForestRegressor(random_state=42) for _ in range(y_train.shape[1])]
    mse_scores = []
    r2_scores = []
    for i, model in enumerate(models):
        # Train each model on the corresponding target
        model.fit(X_train, y_train.iloc[:, i])
        # Make predictions
        predictions = model.predict(X_test)
        # Evaluate and store the MSE for each model
        mse, r2 = compute_model_performance(y_test.iloc[:, i], predictions)
        mse_scores.append(mse)
        r2_scores.append(r2)

    # Overall evaluation
    average_mse = np.mean(mse_scores)
    average_r2 = np.mean(r2_scores)
    model = pred_test_data(test_data_path, model)
    return average_mse, average_r2,model

def Random_Forest_Reg(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[float, float, RandomForestRegressor]:
    def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestRegressor:
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf_model = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

        random_search.fit(X_train, y_train)
        best_rf_model = random_search.best_estimator_
        
        print("Best hyperparameters from RandomizedSearchCV:", random_search.best_params_)

        return best_rf_model
    rf_model = hyperparameter_tuning(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = rf_model.predict(X_test)

    # Compute model performance
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rf_model = pred_test_data(test_data_path, rf_model)
    return mse, r2, rf_model

def Random_Forest_Regressor_hyp(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[float, float, RandomForestRegressor]:
    # Define the base model
    base_model = RandomForestRegressor()

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                               cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    best_hyperparameters = grid_search.best_params_
    print(f"Best Hyperparameters: {best_hyperparameters}")

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions on the test data
    y_pred_test = best_model.predict(X_test)

    # Compute the metrics
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    best_model = pred_test_data(test_data_path, best_model)
    return mse, r2, best_model

def CONV1D_hybrid_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):

    # Create and train the model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(input_column))  # Assuming y_train is your target variable
    var = 1e-5
    learning_rate = 0.1
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    learning_rate = 0.01
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    mse, r2 = compute_model_performance(y_test, predictions)
    conv1d_pred = pred_test_data(test_data_path, model)
    _,_,rf_pred= Random_Forest_Regressor(X_train, X_test, y_train, y_test)
    conv1d_hybrid_prediction=(var*conv1d_pred+(1-var)*rf_pred)/2
    return mse, r2, conv1d_hybrid_prediction

def LSTM_hybrid_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
    # LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=35, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm_model.add(LSTM(units=14, return_sequences=True))  # Last LSTM layer does not return sequences
    lstm_model.add(LSTM(units=10))  # Last LSTM layer does not return sequences
    lstm_model.add(Dense(units=input_column))  
    var = 1e-5
    learning_rate = 0.1  # Set your desired learning rate here
    adam_optimizer = Adam(learning_rate=learning_rate)
    lstm_model.compile(optimizer=adam_optimizer , loss='mean_squared_error')
    # Reshape data for LSTM 
    X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
    lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)
    
    learning_rate_2 = 0.01
    adam_optimizer_2 = Adam(learning_rate=learning_rate_2)
    lstm_model.compile(optimizer=adam_optimizer_2, loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)
    
    learning_rate_3 = 0.001
    adam_optimizer_3 = Adam(learning_rate=learning_rate_3)
    lstm_model.compile(optimizer=adam_optimizer_3, loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)
    lstm_predictions = lstm_model.predict(X_test_lstm)
    lstm_pred = pred_test_data_lstm(test_data_path, lstm_model)
    #RandomFores algorithm
    # Initialize and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    # Make predictions and evaluate the model
    predictions_rf = model.predict(X_test)
    #mse, r2 = compute_model_performance(y_test, predictions)  # Use the helper function for performance printout
    # Overall evaluation
    rf_pred = pred_test_data(test_data_path, model)
    combined_pred_data = (var*lstm_predictions+(1-var)*predictions_rf)
    average_mse = mean_squared_error(y_test, combined_pred_data)
    average_r2 = r2_score(y_test, combined_pred_data)
    lstm_hybrid_predictions = (var*lstm_pred+(1-var)*rf_pred)
    return average_mse,average_r2,lstm_hybrid_predictions



def main(model_type: str, print_mse: bool, print_R_square: bool, data_path):
    # Load and preprocess the data
    X_train, X_test, y_train, y_test= call_data(data_path)


    # Choose the model based on the input parameter
    if model_type == "RandomForest":
        mse,r_square,trained_model = Random_Forest_Regressor(X_train, X_test, y_train, y_test)
    elif model_type == "LightGBM":
        mse,r_square,trained_model = train_LGBMRegressor(X_train, X_test, y_train, y_test)
    elif model_type == "XGBoost":
        mse, r_square,trained_model =  train_XGBRegressor(X_train, X_test, y_train, y_test)
    elif model_type == "XGBoost_RandomSearch":
        mse,r_square,trained_model =  train_XGBRegressor_with_RandomSearch(X_train, X_test, y_train, y_test)
    elif model_type == "RandomForest_hyperparam":
        mse,r_square,trained_model = tune_RandomForest_hyperparameters(X_train, X_test, y_train, y_test)
    elif model_type == "SingleTarget":
        mse,r_square,trained_model = Single_target_prediction(X_train,  X_test, y_train, y_test)
    elif model_type == "Rnd_For_Reg":
        mse,r_square, trained_model = Random_Forest_Reg(X_train,X_test,y_train,y_test)
    elif model_type == "RandomForest_Hyparam":
        mse,r_square, trained_model = Random_Forest_Regressor_hyp(X_train, X_test, y_train, y_test)
    elif model_type == "LSTM_Hybrid":
        mse,r_square,trained_model= LSTM_hybrid_model(X_train, X_test, y_train, y_test) 
    elif model_type == "CONV1d_Hybrid":
        mse,r_square,trained_model= CONV1D_hybrid_model(X_train, X_test, y_train, y_test) 
    else:
        raise ValueError("Invalid model type. Choose from 'RandomForest', 'LightGBM', 'XGBoost','XGBoost_RandomSearch', 'SingleTarget'.")
    
    if print_mse:
        print(f"Model: {model_type}, Mean Squared Error: {(mse)}")
    if print_R_square:
        print(f"Model: {model_type}, R2_square: {(r_square)}")
    
    return trained_model

if __name__ == "__main__":
    # Call the main function with the desired model
    ####Choose one of the next models to train your model
    #trained_model_1 = main("RandomForest", print_mse, print_R_square, data_path_input_2_smaller_zero)
    trained_model_1 = main("RandomForest", print_mse, print_R_square, data_path_input_2_zero)
    #"RandomForest" 
    #"LightGBM"
    #"XGBoost"
    #"XGBoost_RandomSearch"
    #"RandomForest_hyperparam"
    #"SingleTarget"
    #"Rnd_For_Reg"
    #"LSTM_Hybrid"
    #"CONV1d_Hybrid"
    save_data = True
    if save_data:
        #save_data_on_excel(trained_model_1,save_data_path_smaller_zero)
        save_data_on_excel(trained_model_1,save_data_path_zero)
    # To use a different model, just change the argument, e.g., "LightGBM", "XGBoost", "SingleTarget"
    
    





