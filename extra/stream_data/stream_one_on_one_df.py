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
from tensorflow.keras import models
import numpy as np

df = pd.DataFrame(columns=['Device', 'EulerX', 'EulerY', 'EulerZ', 'FreeAccX', 'FreeAccY', 'FreeAccZ', 'Timestamp'])

Moves = ['Clap', 'ArmsOut', 'Elbow', 'Jump', 'Guitar']

difficulty = 1

movement_counts = {move: 0 for move in Moves}

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

def stream():
    global start_time

    print("Starting Stream to stop Streaming press SPACE")
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
        timer_start = time.time() - start_time
        print(f"Start timer: {timer_start}")
        while listener.running:
            if xdpcHandler.packetsAvailable():
                current_time = time.time()
                timestamp = current_time - start_time
                for device in xdpcHandler.connectedDots():
                    packet = xdpcHandler.getNextPacket(device.portInfo().bluetoothAddress())
                    if packet.containsOrientation() and packet.containsFreeAcceleration():
                        euler = packet.orientationEuler()
                        freeAcc = packet.freeAcceleration()

                        df.loc[len(df)] = [device.bluetoothAddress(),
                                           euler.x(), euler.y(), euler.z(),
                                           freeAcc[0], freeAcc[1], freeAcc[2],
                                           timestamp]
                predict(df, timer_start, timestamp)


def predict(df, start_time, timestamp):
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

    # Prepare the input data for prediction
    X = df_pivot.values
    X = np.expand_dims(X, axis=0)

    clap1_interval = 6
    guitar1_interval = 8 + clap1_interval
    jump1_interval = 4 + guitar1_interval
    elbow1_interval = 2 + jump1_interval
    armsOut1_interval = 2 + elbow1_interval
    jump2_interval = 4 + armsOut1_interval
    elbow2_interval = 2 + jump2_interval
    armsOut2_interval = 2 + elbow2_interval

    for x in X:
        # Make prediction for the current row
        predictions = model.predict(X)  # Reshape to 2D array
        predicted_class_index = predictions.argmax()
        predicted_class = Moves[predicted_class_index]

        # Check if the timestamp falls within each time interval and count the corresponding movement
        # Calculate elapsed time within each interval
        clock = int(timestamp - start_time)
        if clock <= clap1_interval and predicted_class == 'Clap':
            movement_counts['Clap'] += 1
        elif clap1_interval < clock <= guitar1_interval and predicted_class == 'Guitar':
            movement_counts['Guitar'] += 1
        elif guitar1_interval < clock <= jump1_interval and predicted_class == 'Jump':
            movement_counts['Jump'] += 1
        elif jump1_interval < clock <= elbow1_interval and predicted_class == 'Elbow':
            movement_counts['Elbow'] += 1
        elif elbow1_interval < clock <= armsOut1_interval and predicted_class == 'ArmsOut':
            movement_counts['ArmsOut'] += 1
        elif armsOut1_interval < clock <= jump2_interval and predicted_class == 'Jump':
            movement_counts['Jump'] += 1
        elif jump2_interval < clock <= elbow2_interval and predicted_class == 'Elbow':
            movement_counts['Elbow'] += 1
        elif elbow2_interval < clock <= armsOut2_interval and predicted_class == 'ArmsOut':
            movement_counts['ArmsOut'] += 1
        else:
            print("Not doing the correct move / Done")








def stopStream():
    """
    Stopping all measurements on all devices
    """
    print("Stopping Measurement mode")
    for device in xdpcHandler.connectedDots():
        if not device.stopMeasurement():
            print("Failed to stop measurement mode.")
    print("Stream completed")

if __name__ == "__main__":
    xdpcHandler = XdpcHandler()
    start_time = time.time()
    initialize()
    set_up()
    print("Press ENTER to redo initialization and setup or press SPACE to start STREAMING.")
    def on_press(key):
        if key == keyboard.Key.enter:
            initialize()
            set_up()
            print("Press ENTER to redo initialization and setup or press SPACE to start STREAMING.")
        elif key == keyboard.Key.space:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    stream()
    print("Clap count:", movement_counts['Clap'])
    print("Jump count:", movement_counts['Jump'])
    print("Guitar count:", movement_counts['Guitar'])
    stopStream()
    print("Press ENTER to stop ALL streams or press SPACE to do another stream")

    def on_press(key):
        if key == keyboard.Key.space:
            df.drop(df.index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            stream()
            print("Clap count:", movement_counts['Clap'])
            print("Jump count:", movement_counts['Jump'])
            print("Guitar count:", movement_counts['Guitar'])
            stopStream()
            print("Press ENTER to stop ALL streams or press SPACE to do another stream")
        elif key == keyboard.Key.enter:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    xdpcHandler.cleanup()


