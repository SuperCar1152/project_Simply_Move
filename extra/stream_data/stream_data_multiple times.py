"""
 For this program the SDK of Movella was used.

 Copyright (c) 2003-2023 Movella Technologies B.V. or subsidiaries worldwide.
 All rights reserved.
 Link:

 Adapted by Carlijn le Clercq and Yana Volders
 Data acquisition for XSense Movella DOTs (to CSV Files)
 """

import movelladot_pc_sdk.movelladot_pc_sdk_py310_64

from xdpchandler_SDK import *
import pandas as pd
import time
import datetime
from tensorflow.keras import models
import numpy as np

logName = "testStream"
logMap = "testCSV"

Moves = ['Clap', 'ArmsOut', 'Elbow', 'Jump', 'Guitar']

model = models.load_model("Model/ModelsTrained/trained_model8.keras")

def initialize():
    print("Initializing DOTs")

    # Initialize
    if not xdpcHandler.initialize():
        print("Failed to initialize SDK")

    # Scan for XSense DOTS
    xdpcHandler.scanForDots()
    if len(xdpcHandler.detectedDots()) == 0:
        print("No Movella DOT device(s) found. Aborting.")

    # Connect to detected DOTS
    xdpcHandler.connectDots()
    if len(xdpcHandler.connectedDots()) == 0:
        print("Could not connect to any Movella DOT device(s). Aborting.")

    print(f"Connected a total of DOTs: {len(xdpcHandler.connectedDots())}")


def set_up():
    print("Setting up Devices")

    # Set up for every connected device
    for device in xdpcHandler.connectedDots():
        # Set filter profile
        if not device.setOnboardFilterProfile("General"):
            print("Setting filter profile to General failed!")

        # Set output rate
        if not device.setOutputRate(30):
            print("Setting output rate failed!")

        print(f"Successfully setup {device.bluetoothAddress()}")

def stream():
    global start_time, df

    print("Starting stream to stop stream press SPACE")
    for device in xdpcHandler.connectedDots():
        if not device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler):
            print(f"Could not put device into measurement mode. Reason: {device.lastResultText()}, trying again")
            if not device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler):
                print(f"Again, could not put device into measurement mode. Reason: {device.lastResultText()}")

    def on_press(key):
        if key == keyboard.Key.space:
            return False
        return True

    with keyboard.Listener(on_press=on_press) as listener:
        while listener.running:
            if xdpcHandler.packetsAvailable():
                current_time = time.time()
                timestamp = current_time - start_time

                # Prepare data for appending to DataFrame
                data_row = {'Timestamp': timestamp}
                for device in xdpcHandler.connectedDots():
                    packet = xdpcHandler.getNextPacket(device.portInfo().bluetoothAddress())
                    if packet.containsOrientation() and packet.containsFreeAcceleration():
                        euler = packet.orientationEuler()
                        freeAcc = packet.freeAcceleration()

                        # Append sensor data to the data_row dictionary
                        for sensor_data, value in zip(
                                ['EulerX', 'EulerY', 'EulerZ', 'FreeAccX', 'FreeAccY', 'FreeAccZ'],
                                [euler.x(), euler.y(), euler.z(), freeAcc[0], freeAcc[1], freeAcc[2]]):
                            data_row[f"{device.bluetoothAddress()}_{sensor_data}"] = value

                print("Data row:", data_row)  # Debugging statement

                # Check if a row with the same timestamp already exists
                existing_row_index = df[df['Timestamp'] == timestamp].index
                if len(existing_row_index) == 0:
                    # Append data_row to DataFrame
                    df = pd.concat([df, pd.DataFrame(data_row, index=[0])], ignore_index=True)
                else:
                    existing_row_index = existing_row_index[0]
                    existing_row = df.loc[existing_row_index]
                    if existing_row.isna().any():  # Check if the row is incomplete
                        # Update the existing row with new data
                        df.loc[existing_row_index] = data_row
                    else:
                        # Append data_row to DataFrame
                        df = pd.concat([df, pd.DataFrame(data_row, index=[0])], ignore_index=True)

                # Convert data_row to DataFrame
                df_temp = pd.DataFrame(data_row, index=[0])

                # Ensure each sequence has length 97
                df_temp = df_temp.reindex(range(97), fill_value=0)  # Pad with zeros if sequence is shorter than 97

                # Ensure each sequence has 30 features
                df_temp = df_temp.iloc[:, :30]  # Truncate if sequence has more than 30 features

                # Reshape DataFrame to match model input shape
                df_temp = df_temp.to_numpy().reshape(1, 97, 30)

                # Make predictions
                predictions = model.predict(df_temp)
                predicted_class_index = np.argmax(predictions)
                predicted_class = Moves[predicted_class_index]

                # Print predicted class and probability of predicted class
                predicted_class_prob = predictions[0][predicted_class_index]
                print("Predicted class:", predicted_class)
                print(f"Probability for {predicted_class}: {predicted_class_prob:.4f}")
                print()


def stopStream():
    """
    Stopping all measurements on all devices
    """
    print("Stopping stream")
    for device in xdpcHandler.connectedDots():
        if not device.stopMeasurement():
            print("Failed to stop measurement mode.")
    print("Data acquisition complete, stopped stream")

def df_to_csv(mapLog, nameLog):
    print("Saving dataframe to CSV")

    # Define the file path
    file_path = f"{mapLog}/{nameLog}.csv"

    # Save the DataFrame to CSV
    if not df.to_csv(file_path, index=False):
        print("Could not save dataframe to CSV")

    print(f"DataFrame saved to {file_path}")

def predict(frame):
    frame += 2
    return frame


if __name__ == "__main__":
    xdpcHandler = XdpcHandler()
    start_time = time.time()
    initialize()
    set_up()
    print("Press ENTER to redo initialization and setup or press SPACE to start measuring.")
    def on_press(key):
        if key == keyboard.Key.enter:
            initialize()
            set_up()
            print("Press ENTER to redo initialization and setup or press SPACE to start measuring.")
        elif key == keyboard.Key.space:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # Initialize DataFrame columns dynamically based on connected devices
    df_columns = ['Timestamp']
    for device in xdpcHandler.connectedDots():
        for sensor_data in ['EulerX', 'EulerY', 'EulerZ', 'FreeAccX', 'FreeAccY', 'FreeAccZ']:
            df_columns.append(f"{device.bluetoothAddress()}_{sensor_data}")

    df = pd.DataFrame(columns=df_columns)

    stream()
    stopStream()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df_to_csv(mapLog=logMap, nameLog=f"{logName}_{current_time}")
    print("Press ENTER to stop ALL streams or press SPACE to do another stream")

    def on_press(key):
        if key == keyboard.Key.space:
            df.drop(df.index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            stream()
            stopStream()
            current_time2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            df_to_csv(mapLog=logMap, nameLog=f"{logName}_{current_time2}")
            print("Press ENTER to stop ALL streams or press SPACE to do another stream")
        elif key == keyboard.Key.enter:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    xdpcHandler.cleanup()
