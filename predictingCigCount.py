import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

######################################################################################################################################################
#changing data

df = pd.read_csv('data/smoking_health_data_final.csv')

# print(df.head(10))

## only first 13 so we replace with 0
# for index, row in df.iterrows():
#     if pd.isna(row['cigs_per_day']):
#         print("Index:", index, "Value:", row['cigs_per_day'])
df['cigs_per_day'] = df['cigs_per_day'].fillna(0)

# print(df.isnull().sum())
# Impute missing values in the 'chol' column with the mean
df['chol'].fillna(df['chol'].mean(), inplace=True)

# Split the "blood_pressure" column into two separate columns: "systolic_pressure" and "diastolic_pressure"
df[['systolic_pressure', 'diastolic_pressure']] = df['blood_pressure'].str.split('/', expand=True)

# Convert the new columns to numeric type
df['systolic_pressure'] = pd.to_numeric(df['systolic_pressure'], errors='coerce')
df['diastolic_pressure'] = pd.to_numeric(df['diastolic_pressure'], errors='coerce')

# Drop the original "blood_pressure" column
df.drop(columns=['blood_pressure'], inplace=True)

encoded_categorical_data = pd.get_dummies(df['sex'], prefix='sex')
df = pd.concat([df, encoded_categorical_data], axis=1)

print(df.head(100))

######################################################################################################################################################
#changing data but on features (can mix and match with prev bit kinda)

categ_feat = ['sex_male', 'sex_female']

numeric_feat = ['age', 'heart_rate', 'systolic_pressure', 'diastolic_pressure', 'chol']

scaler = StandardScaler()
df[numeric_feat] = scaler.fit_transform(df[numeric_feat])

######################################################################################################################################################
# get final target and features

target_col = 'cigs_per_day'

feature_cols = ['age', 'heart_rate', 'systolic_pressure', 'diastolic_pressure', 'chol', 'sex_male', 'sex_female']

for i in feature_cols:
    print(i)
######################################################################################################################################################

# shuffle df
df_shuffled = df.sample(frac=1, random_state=42)

train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# Split the data into train and test sets first, then to validation
X_train_val, X_test, y_train_val, y_test = train_test_split(df_shuffled[feature_cols], df_shuffled[target_col], test_size=test_ratio, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_ratio/(train_ratio+validation_ratio), random_state=42)

print("Unique values in y_train:", np.unique(y_train))

print("Feature statistics:")
print(X_train.describe())

######################################################################################################################################################

# Model building
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1)  # For regression task, no activation function in the last layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

######################################################################################################################################################
# use the model

#train 
history = model.fit(X_train, y_train, epochs=60, batch_size=8, validation_data=(X_val, y_val))

# Evaluate the model on the testing set
test_loss, test_mae = model.evaluate(X_test, y_test)

# Print test metrics
print('Test Loss:', test_loss)
print('Test MAE:', test_mae)

######################################################################################################################################################
# Visualize training
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
# Save the plot as a file
plt.savefig('cigs_prediction.png', format='png')
plt.close()  # Close the plot to free up memory

######################################################################################################################################################
#print results

# Predict on the test set
y_pred = model.predict(X_test)

# Create a DataFrame to store actual and predicted values
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred.flatten()})

# Print the first few rows of the results DataFrame
print(results_df.head(50))

######################################################################################################################################################
#using the model

def predict_cigarettes_per_day(age, heart_rate, systolic_pressure, diastolic_pressure, chol, sex_male, sex_female):
    # Create a DataFrame with the input data
    custom_data = pd.DataFrame({
        'age': [age],
        'heart_rate': [heart_rate],
        'systolic_pressure': [systolic_pressure],
        'diastolic_pressure': [diastolic_pressure],
        'chol': [chol],
        'sex_male': [sex_male],
        'sex_female': [sex_female]
    })
    
    # Normalize numerical features using the same scaler used for training data
    custom_data[numeric_feat] = scaler.transform(custom_data[numeric_feat])

    # Use the trained model to predict the number of cigarettes per day for the custom input data
    prediction = model.predict(custom_data)
    
    # Return the predicted number of cigarettes per day
    return prediction[0][0]

# Example usage:
predicted_cigarettes = predict_cigarettes_per_day(age=42, heart_rate=75, systolic_pressure=120, diastolic_pressure=80, chol=200, sex_male=1, sex_female=0)
print('Predicted number of cigarettes per day:', predicted_cigarettes)