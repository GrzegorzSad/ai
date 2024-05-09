import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#fuck it i cant make the gpu work, i doesnt matter anyway theese fucks get done on google cloud anyway


# Loading the dataset
data = pd.read_csv('data/2022_Green_Taxi_Trip_Data.csv')

##FORMATTING THE DATA

# Check for missing values
# print(data.isnull().sum())

# Handle missing values (here replaced with 0 cos thats better xd)
data.fillna(0, inplace=True)

# print(data.isnull().sum())

# Convert datetime columns
data['lpep_pickup_datetime'] = pd.to_datetime(data['lpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
data['lpep_dropoff_datetime'] = pd.to_datetime(data['lpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# Create new features
data['pickup_hour'] = data['lpep_pickup_datetime'].dt.hour
data['day_of_week'] = data['lpep_pickup_datetime'].dt.dayofweek
data['trip_duration'] = (data['lpep_dropoff_datetime'] - data['lpep_pickup_datetime']).dt.total_seconds() / 60  # duration in minutes

# Convert categorical variables using one-hot encoding (aka 0 or 1)
categorical_features = ['VendorID', 'store_and_fwd_flag', 'RatecodeID']
data = pd.get_dummies(data, columns=categorical_features)

# Select features (remove columns that are not to be used)
features_to_drop = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'ehail_fee']
data.drop(columns=features_to_drop, inplace=True)

# Define your target variable
target = 'fare_amount'

# Define your predictors (all columns that are not the target)
predictors = [col for col in data.columns if col != target]

#split data, here with numpy, could have also been done with sklearn, whatevs
train, val, test = np.split(data.sample(frac=1), [int(0.8*len(data)), int(0.9*len(data))])

print('data:')
print(data.head(10))
print(len(train), 'training examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#BUILDING THE MODEL

# Create a Sequential model
model = Sequential([
    Dense(32, activation='relu', input_shape=(train[train.columns.difference(['fare_amount'])].shape[1],)),  # Make sure to exclude the target variable
    Dense(8, activation='relu', name='h1'),
    Dense(8, activation='relu', name='h2'),
    Dense(1, activation='linear', name='h3')  # Linear activation for the output layer
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
print('model summary:')
model.summary()


#TRAIN THE MODEL

# Extract features and target from train, val, and test
X_train = train[predictors]
y_train = train[target]

X_val = val[predictors]
y_val = val[target]

X_test = test[predictors]
y_test = test[target]


# Set up early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # Output messages
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)

#model.fit trains it
history = model.fit(
    x=X_train, 
    y=y_train, 
    epochs=7,  # Example number of epochs
    batch_size=32,  # Example batch size
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]  # Include the early stopping callback here
)

# Visualize training
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
# Save the plot as a file
plt.savefig('training_validation_loss.png', format='png')
plt.close()  # Close the plot to free up memory

#EVALUATE
test_loss = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)

predictions = model.predict(X_test)

# Print the first 50 actual values and their corresponding predictions
for actual, predicted in zip(y_test[:50], predictions.flatten()[:50]):
    print(f"Actual: {actual:.2f}, Predicted: {predicted:.2f}")

