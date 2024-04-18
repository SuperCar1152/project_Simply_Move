"""
 For this program the SDK of Movella was used.

 Copyright (c) 2003-2023 Movella Technologies B.V. or subsidiaries worldwide.
 All rights reserved.
 Link:

 Adapted by Carlijn le Clercq and Yana Volders
 Data acquisition for XSense Movella DOTs (to CSV Files)
 """

import movelladot_pc_sdk.movelladot_pc_sdk_py310_64

from xdpchandler import *
import pandas as pd
import time
import datetime
from tensorflow.keras import models
import numpy as np

logName = "testStream"
logMap = "testCSV"

Moves = ['Clap', 'ArmsOut', 'Elbow', 'Jump', 'Guitar']
model = models.load_model('Model/ModelsTrained/trainedYANAAAAA.keras')

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

def measurement():
    global start_time

    print("Starting measurement to stop measuring press SPACE")
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

                # Convert data_row to NumPy array
                data_array = np.array([list(data_row.values())])

                # Reshape the data_array to match the expected input shape of the model
                num_devices = len(xdpcHandler.connectedDots())
                num_features = len(data_row) - 1  # Excluding Timestamp
                data_array_reshaped = data_array.reshape((1, num_devices, num_features))

                print("Data row:", data_row)  # Debugging statement
                print("Data array:", data_array_reshaped)  # Debugging statement

                # Make predictions
                predictions = model.predict(data_array)
                predicted_class_index = np.argmax(predictions)
                predicted_class = Moves[predicted_class_index]

                # Print predicted class and probabilities for each class
                print("Predicted class:", predicted_class)
                for move, prob in zip(Moves, predictions[0]):
                    print(f"Probability for {move}: {prob:.4f}")
                print()


def stopMeasurement():
    """
    Stopping all measurements on all devices
    """
    print("Stopping Measurement")
    for device in xdpcHandler.connectedDots():
        if not device.stopMeasurement():
            print("Failed to stop measurement.")
    print("Data acquisition complete, stopped measurement")

def df_to_csv(mapLog, nameLog):
    print("Saving dataframe to CSV")

    # Define the file path
    file_path = f"{mapLog}/{nameLog}.csv"

    # Save the DataFrame to CSV
    if not df.to_csv(file_path, index=False):
        print("Could not save dataframe to CSV")

    print(f"DataFrame saved to {file_path}")

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

    measurement()
    stopMeasurement()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df_to_csv(mapLog=logMap, nameLog=f"{logName}_{current_time}")
    print("Press ENTER to stop measurements or press SPACE to do another measurement")

    def on_press(key):
        if key == keyboard.Key.space:
            df.drop(df.index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            measurement()
            stopMeasurement()
            current_time2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            df_to_csv(mapLog=logMap, nameLog=f"{logName}_{current_time2}")
            print("Press ENTER to stop measurements or press SPACE to do another measurement")
        elif key == keyboard.Key.enter:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    xdpcHandler.cleanup()


