"""
Data Preprocessing and LSTM Model Training for Human Movement Recognition

This code preprocesses accelerometer and gyroscope data collected from wearable sensors during various human movements,
such as Clapping, Extending Arms, Elbow Movement, Playing Guitar, and Jumping.
The preprocessed data is then used to train an LSTM (Long Short-Term Memory)
neural network model for classifying these movements.

The preprocessing steps include:
1. Reading raw sensor data from CSV files.
2. Sorting the data based on timestamp.
3. Reshaping and cleaning the data.
4. Saving the modified data to new CSV files.

After preprocessing, the data from the modified CSV files is combined and split into training, validation,
and testing sets. The LSTM model architecture is defined and compiled, with early stopping implemented to prevent
overfitting. The model is trained on the training data and evaluated using the validation set.
Finally, the trained model is saved, and its performance is assessed using various metrics including
accuracy, loss, classification report, and confusion matrix.

Note: Ensure that the directory structure matches the specified paths for data loading and saving.

Written by: Carlijn le Clercq and Yana Volders
"""
import pandas as pd
import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, LSTM
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers, Sequential
import os
import seaborn as sns
import matplotlib.pyplot as plt

print('Start')

# Directory containing CSV files for each move type
datadir_train = 'dataCSV'

# List of move types
Moves = ['Clap', 'ArmsOut', 'Elbow', 'Guitar', 'Jump']

# Loop through each move type
for move in Moves:
    print(f'Loading data for move: {move}')
    move_dfs = []
    path = os.path.join(datadir_train, move)

    # Loop through each CSV file in the move directory
    for file in os.listdir(path):
        if file.endswith(".csv"):
            # Read CSV file into a DataFrame
            df = pd.read_csv(os.path.join(path, file))

            # Sort the DataFrame based on the 'Timestamp' column
            df.sort_values(by='Timestamp', inplace=True)

            # Pivot the DataFrame
            df_pivot = df.pivot(index='Timestamp', columns='Device')

            # Reorder the MultiIndex columns to have the sensor data columns grouped by sensor and then by measurement type
            df_pivot.columns = df_pivot.columns.swaplevel(0, 1)
            df_pivot.sort_index(axis=1, level=[0, 1], inplace=True)

            # Flatten MultiIndex columns
            df_pivot.columns = ['_'.join(col) for col in df_pivot.columns]

            # Reset index to convert Timestamp to a regular column
            df_pivot.reset_index(inplace=True)

            # Remove records with missing values
            df_pivot.dropna(inplace=True)

            # Sort column on name
            df_pivot.reindex(sorted(df.columns), axis=1)

            # Write the modified DataFrame to a new CSV file
            df_pivot.to_csv(f'Model/Data/modified/{move}/{file}modified_file.csv', index=False)

# Combine Data from Modified CSV Files
X_data = []
y_data = []

modified_dir = 'Model/Data/modified'

# Initialize an empty DataFrame to hold all the data
combined_df = pd.DataFrame()

# Loop through each modified CSV file in the move directory
for move in Moves:
    print(f'Loading modified data for move: {move}')
    path = os.path.join(modified_dir, move)

    # Loop through each modified CSV file in the move directory
    for file in os.listdir(path):
        if file.endswith("modified_file.csv"):
            # Read modified CSV file into a DataFrame
            ddf = pd.read_csv(os.path.join(path, file))

            # Add a label column with the move name
            ddf['Label'] = move

            # Concatenate the DataFrame to the combined DataFrame
            combined_df = pd.concat([combined_df, ddf], ignore_index=True)

# Specify the path to save the CSV file
output_csv_path = 'Model/Data/combinedAllData.csv'

# Save the DataFrame to a CSV file
combined_df.to_csv(output_csv_path, index=False)

print(f'Combined data saved to {output_csv_path}')

label_map = {'Clap': 0, 'ArmsOut': 1, 'Elbow': 2, 'Guitar': 3, 'Jump': 4}

# Separate features (X)
X = combined_df.drop(columns=['Label'])  # All data except the 'Label' column

# Convert labels to numerical values using label_map
y = np.array([label_map[label] for label in combined_df['Label']])

# Verify the shapes of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Number of labels assigned:", len(y))
print('X,y made')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print('Data splitted')

# Reshape X_train and X_test to include timestep dimension
X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val = X_val.values.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(units=20, return_sequences=True, input_shape=X_train.shape[1:]))
model.add(LSTM(units=20, return_sequences=True))
model.add(LSTM(units=20))
model.add(Dense(5, activation='softmax'))
print('Model made')

optimizer = optimizers.Adam(learning_rate=0.001)

# Compile the Model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Model compiled')

model.summary()

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the Model
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test),
                    callbacks=[early_stopping])
print('Model trained')

# Evaluate the Model
loss, accuracy = model.evaluate(X_val, y_val)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the Model
model.save('Model/ModelsTrained/trainedYANAAAAA4.keras')


def plot_learning_curves(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# Assuming 'history' contains training history returned by model.fit()
plot_learning_curves(history)

# Make Predictions
y_pred = model.predict(X_test)
model.summary()

# Make predictions
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate classification report
report = classification_report(y_test, y_pred_classes)

# Print the classification report
print(report)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"{move}" for move in Moves],
            yticklabels=[f"{move}" for move in Moves])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
