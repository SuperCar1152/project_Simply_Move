import ast
import csv
import json
import pandas as pd
import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, LSTM
from sklearn import metrics
from keras.src.utils import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

modified_dir = '../../Model/Data/modified'

data_dict = {
    'timestamp': [],
    'label': [],
    'frame': [],
    'separator': []
}

# Loop through each modified CSV file in the move directory
for move in Moves:
    print(f'Loading modified data for move: {move}')
    path = os.path.join(modified_dir, move)

    # Loop through each modified CSV file in the move directory
    for file in os.listdir(path):
        if file.endswith("modified_file.csv"):
            # Read modified CSV file into a DataFrame
            df = pd.read_csv(os.path.join(path, file))

            # Extract timestamp, label, and frame data
            timestamp = df['Timestamp'].tolist()
            label = [move] * len(timestamp)
            frame = df[df.columns[df.columns.str.contains('Euler|FreeAcc')]].values.tolist()

            # Set separator to True for the first entry, False for the rest
            separator = [True] + [False] * (len(timestamp) - 1)

            # Append the data to the dictionary
            data_dict['timestamp'].extend(timestamp)
            data_dict['label'].extend(label)
            data_dict['frame'].extend(frame)
            data_dict['separator'].extend(separator)

#
# Verify the data
for key, value in data_dict.items():
    print(f'{key}: {value[:5]}')  # Display first 5 entries of each key

def csv_to_json(csv_file, json_file):
    data = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)

    with open(json_file, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)

# Convert the dictionary to a DataFrame
ddf = pd.DataFrame(data_dict)

# Specify the path to save the CSV file
output_csv_path = '../../Model/Data/combined_data_corrected.csv'

# Save the DataFrame to a CSV file
ddf.to_csv(output_csv_path, index=False)

print(f'Combined data saved to {output_csv_path}')

# Usage
csv_to_json(output_csv_path, '../../Model/data.json')


with open("../../Model/data.json", 'r') as file:
    Xsensors = json.load(file)

# Initialize lists to store sequences of frames and corresponding labels
X_sequences = []
y_labels = []

# Initialize variables to store current sequence and label
current_sequence = []
current_label = None

# Iterate through the data
for entry in Xsensors:
    # Check if separator is true, indicating the start of a new movement
    if entry["separator"] == "True":
        # If it's not the first movement, append the current sequence and label
        if current_label is not None:
            X_sequences.append(current_sequence)
            y_labels.append(current_label)
        # Start a new sequence for the current movement
        current_sequence = [ast.literal_eval(entry["frame"])]
        current_label = entry["label"]
    else:
        # Continue the current sequence for the same movement
        current_sequence.append(ast.literal_eval(entry["frame"]))

    # Append the last sequence and label
if current_label is not None:
    X_sequences.append(current_sequence)
    y_labels.append(current_label)

# # Pad sequences to ensure uniform length
X_padded = pad_sequences(X_sequences, padding='post', dtype='float32')

# # Convert string values to floats
# X_padded = np.array([[list(map(float, frame)) for frame in sequence] for sequence in X_padded])
#
# # Initialize standardScaler
# scaler = MinMaxScaler()
#
# # Scale each sequence individually
# X_scaled = np.array([scaler.fit_transform(seq) for seq in X_padded])

label_map = {'Clap': 0, 'ArmsOut': 1, 'Elbow': 2, 'Guitar': 3, 'Jump': 4}

# Convert lists to arrays
X = np.array(X_padded)
y = np.array([label_map[label] for label in y_labels])

# Verify the shapes of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Number of labels assigned:", len(y))
print('X,y made')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
print('data splitted')


model = Sequential()
model.add(LSTM(units=15, return_sequences=True, input_shape=X_train.shape[1:]))
model.add(LSTM(units=15, return_sequences=True))
model.add(LSTM(units=10))
model.add(Dense(4, activation='softmax'))

print('Model made')

optimizer = optimizers.Adam(learning_rate=0.001)

# Compile the Model
model.compile(optimizer= optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Model compiled')

model.summary()

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the Model
history = model.fit(X_train, y_train, epochs=200, batch_size=6, validation_data=(X_test,y_test), callbacks=[early_stopping])
print('Model trained')

# Evaluate the Model
loss, accuracy = model.evaluate(X_val, y_val)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

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

model.save('Model/ModelsTrained/trained_model.keras')

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
