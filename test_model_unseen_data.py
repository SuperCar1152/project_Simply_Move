"""
Predict Human Movement Types Using Pretrained LSTM Model

This program utilizes a pretrained LSTM (Long Short-Term Memory) neural network model to predict human movement types
from accelerometer and gyroscope data collected from wearable sensors. The model is loaded from a pre-existing file
and applied to new data obtained from CSV files.

Functions:
- predict_movement_types(data_dir, model, Moves): Predicts human movement types from data in the specified
directory using the provided model and movement types.

Usage:
1. Ensure the trained model file ('trained_model8.keras') and the data directory ('data') are correctly specified.
2. Run the script to predict movement types from the data files.
3. Predicted data is saved to 'predicted_data.csv'.
4. Predicted classes and their probabilities are printed for each file processed.

Written by Carlijn le Clerca and Yana Volders
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras import models

Moves = ['Clap', 'ArmsOut','Elbow', 'Guitar', 'Jump']
model = models.load_model('Model/ModelsTrained/trained_model8.keras')

data_dir = 'Model/AAAAA/data'

print('Start:)')
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        # Read CSV file into a DataFrame
        df = pd.read_csv(os.path.join(data_dir, file))

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

        # Save df_pivot to a CSV file
        output_csv_path = 'predicted_data.csv'
        df_pivot.to_csv(output_csv_path, index=False)
        print(f'Predicted data saved to {output_csv_path}')

        # Prepare the input data for prediction
        X = df_pivot.values
        X = np.expand_dims(X, axis=0)

        # Make predictions for each row in X_data
        predicted_classes = []
        for x in X:
            # Make prediction for the current row
            prediction = model.predict(X)  # Reshape to 2D array
            predicted_class_index = prediction.argmax()
            predicted_class = Moves[predicted_class_index]
            predicted_classes.append(predicted_class)

            # Print predicted class and probabilities for each class
            print("File:", file)
            print("Predicted class:", predicted_class)
            for move, prob in zip(Moves, prediction[0]):
                print(f"Probability for {move}: {prob:.4f}")