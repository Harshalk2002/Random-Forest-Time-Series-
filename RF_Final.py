import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the dataset with dtype parameter
data = pd.read_csv("session_details.csv", encoding='latin1', dtype={'LAST_UPDATED_DATE': str}, low_memory=False)

# Convert LAST_UPDATED_DATE to datetime format with errors='coerce' to handle errors gracefully
data['LAST_UPDATED_DATE'] = pd.to_datetime(data['LAST_UPDATED_DATE'], format='%m/%d/%Y %H:%M', errors='coerce')



# Aggregate users per week
data['Week'] = data['LAST_UPDATED_DATE'].dt.to_period('W')
users_per_week = data.groupby('Week')['ID'].nunique()

# Calculate lag for users per week
users_per_week_lag = users_per_week.shift(1)

# Combine users per week and lag into a DataFrame
users_per_week_with_lag = pd.DataFrame({'Week': users_per_week.index,
                                        'Users_Per_Week': users_per_week.values,
                                        'Users_Per_Week_Lag': users_per_week_lag.values})

# Print the DataFrame with users per week and lag
print(users_per_week_with_lag)




# Create a list to store the projections for each week
weekly_projections = []

# Iterate over each week
for week_num in range(17):
    # Calculate the projected users per week for the current week using the lag
    if week_num == 0:
        # For the first week, use the actual users per week as projection
        projection = users_per_week_with_lag.iloc[0]['Users_Per_Week']
    else:
        # For subsequent weeks, use the lagged users per week
        lagged_users_per_week = users_per_week_with_lag.iloc[week_num - 1]['Users_Per_Week_Lag']
        if pd.isnull(lagged_users_per_week):
            # If lagged value is NaN, use the actual value of the first week
            projection = users_per_week_with_lag.iloc[0]['Users_Per_Week']
        else:
            # Assuming a growth rate factor of 1.05 (5% increase)
            projection = lagged_users_per_week * 1.05

    # Append the projection to the list
    week_start_date = users_per_week_with_lag.iloc[week_num]['Week'].start_time.strftime('%Y-%m-%d')
    week_end_date = users_per_week_with_lag.iloc[week_num]['Week'].end_time.strftime('%Y-%m-%d')
    week_period = f"{week_start_date}/{week_end_date}"
    weekly_projections.append({'Week': week_period, 'Projected_Users_Per_Week': projection})

# Convert the list of dictionaries to a DataFrame
weekly_projections_df = pd.DataFrame(weekly_projections)

# Print the DataFrame with weekly projections
print(weekly_projections_df)





# Drop rows with NaN values in LAST_UPDATED_DATE
data = data.dropna(subset=['LAST_UPDATED_DATE'])

# Convert 'TOTAL_TOKEN' to numeric type, coerce errors to NaN
data['TOTAL_TOKEN'] = pd.to_numeric(data['TOTAL_TOKEN'], errors='coerce')

# Drop rows with NaN values in 'TOTAL_TOKEN'
data = data.dropna(subset=['TOTAL_TOKEN'])

# Aggregate total tokens per hour
data['Hour'] = data['LAST_UPDATED_DATE'].dt.floor('h')
data = data.groupby('Hour').agg({'TOTAL_TOKEN': 'sum'}).reset_index()

# Feature Engineering
data['hour_of_day'] = data['Hour'].dt.hour
data['day_of_week'] = data['Hour'].dt.dayofweek
data['day_of_month'] = data['Hour'].dt.day
data['month'] = data['Hour'].dt.month
data['year'] = data['Hour'].dt.year
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Define features and target variable
X = data[['hour_of_day', 'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend']]
y = data['TOTAL_TOKEN']

# Include lag of 1 week
data['TOTAL_TOKEN_lag_week'] = data['TOTAL_TOKEN'].shift(168)

# Split data using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['hour_of_day', 'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend']),
        ('pca', PCA(n_components=5), ['hour_of_day', 'day_of_week', 'day_of_month', 'month', 'year'])
    ])

# Define the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(random_state=42))])

# Define hyperparameters grid for Random Forest Regressor
param_grid = {
    'regressor__n_estimators': [300],  # Number of trees in the forest
    'regressor__max_depth': [600],  # Maximum depth of the tree
    'regressor__min_samples_split': [20],  # Minimum number of samples required to split a node
    'regressor__min_samples_leaf': [20],  # Minimum number of samples required at each leaf node
    #'regressor__max_features': ['auto', 'sqrt']  # Number of features to consider when looking for the best split
}

# Initialize GridSearchCV with the hyperparameters grid
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Fit the best model on the train data
best_model.fit(X_train, y_train)

# Predict on the test set using the best model
y_pred = best_model.predict(X_test)

# Calculate MAPE (Mean Absolute Percentage Error)
mape = (1 / len(y_test)) * np.sum(np.abs((y_test - y_pred) / y_test)) * 100
print("Mean Absolute Percentage Error (MAPE):", mape)

# Predict for the entire duration of the dataset using the model with best hyperparameters
first_hour = data['Hour'].min()
last_hour = data['Hour'].max()
all_hours = pd.date_range(start=first_hour, end=last_hour, freq='h')
all_data = pd.DataFrame({'Hour': all_hours,
                         'hour_of_day': all_hours.hour,
                         'day_of_week': all_hours.dayofweek,
                         'day_of_month': all_hours.day,
                         'month': all_hours.month,
                         'year': all_hours.year,
                         'is_weekend': ((all_hours.dayofweek) // 5 == 1).astype(int)})

# Include lag of 1 week starting from Monday
all_data['TOTAL_TOKEN_lag_week'] = all_data['Hour'].dt.floor('d') - timedelta(days=7)
all_data['TOTAL_TOKEN_lag_week'] = all_data.merge(data[['Hour', 'TOTAL_TOKEN']], how='left', left_on='TOTAL_TOKEN_lag_week', right_on='Hour')['TOTAL_TOKEN']
all_data['TOTAL_TOKEN_lag_week'] = all_data.groupby(all_data['Hour'].dt.dayofweek)['TOTAL_TOKEN_lag_week'].transform(lambda x: x.ffill())

# Convert 'Hour' column to the appropriate data type
all_data['Hour'] = pd.to_datetime(all_data['Hour'])

predicted_tokens = best_model.predict(all_data[['hour_of_day', 'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend', 'TOTAL_TOKEN_lag_week']])

# Convert the predicted token counts to a DataFrame
predicted_df = pd.DataFrame({'Hour': all_hours, 'Predicted_Token_Count': predicted_tokens})

# Merge actual and predicted token counts into a single DataFrame
combined_df = pd.merge(data[['Hour', 'TOTAL_TOKEN', 'TOTAL_TOKEN_lag_week']], predicted_df, on='Hour', how='outer')

# Identify the timestamp before the 24-hour prediction window starts
start_time_prediction_window = last_hour - timedelta(hours=24)

# Filter the dataset to include only data points before the prediction window
data_before_prediction_window = data[data['Hour'] < start_time_prediction_window]

# Calculate the total number of users before the prediction window
total_users_before_prediction = data_before_prediction_window['TOTAL_TOKEN'].sum()

print("Total number of users before the 24-hour prediction:", total_users_before_prediction)

# Extend the time series to include the next 24 hours
next_24_hours_data = pd.DataFrame({'Hour': pd.date_range(start=last_hour + timedelta(hours=1), periods=24, freq='h')})
next_24_hours_data['TOTAL_TOKEN'] = np.nan  # Fill with NaN as we don't have actual values for the next 24 hours

# Concatenate the actual data and the next 24 hours data
extended_data = pd.concat([data, next_24_hours_data])

# Plotting actual, predicted, and lagged token counts for the past three months and the next 24 hours
plt.figure(figsize=(16, 6))

# Plot actual token counts for the past three months
plt.plot(extended_data['Hour'], extended_data['TOTAL_TOKEN'], color='blue', label='Actual', linestyle='-')

# Plot predicted token counts for the entire duration
plt.plot(predicted_df['Hour'], predicted_df['Predicted_Token_Count'], color='red', label='Predicted', linestyle='-')

# Plot lagged token counts for the entire duration
plt.plot(extended_data['Hour'], extended_data['TOTAL_TOKEN_lag_week'], color='green', label='1 week lag', linestyle='-')

# Setting labels and title
plt.title('Actual vs Predicted vs Lagged Token Counts (Past Three Months and Predictions)')
plt.xlabel('Hour')
plt.ylabel('Token Count')

# Adding legend, grid, and rotating x-axis labels for better readability
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Append users_per_week_with_lag to CSV file
users_per_week_with_lag.to_csv('prediction.csv', mode='a', header=True, index=False)

# Append weekly_projections_df to CSV file
weekly_projections_df.to_csv('prediction.csv', mode='a', header=True, index=False)




