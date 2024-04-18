"""
Data Acquisition for XSense Movella DOTs to CSV Files

This program utilizes the SDK provided by Movella Technologies B.V. to acquire data from XSense Movella DOTs
(Inertial Measurement Units) and save it to CSV files.

Functions:
- initialize(): Initializes the XSense DOTs, scans for connected devices, and establishes connections.
- set_up(): Sets up filter profiles and output rates for connected devices.
- measurement(): Initiates data measurement from connected devices and records orientation Euler angles and free accelerations.
- stopMeasurement(): Stops data measurement on all connected devices.
- df_to_csv(mapLog, nameLog): Saves the recorded data in a DataFrame to a CSV file.

Usage:
1. Press ENTER to redo initialization and setup or press SPACE to start measuring.
2. Press SPACE to start measuring.
3. Press ENTER to stop measurements or press SPACE to do another measurement.

Note: Ensure the XSense DOTs are properly connected and configured before running the program.

The program is written by Carlijn le Clercq and Yana Volders, using examples from the Movella SDK.
"""
import movelladot_pc_sdk.movelladot_pc_sdk_py310_64

from xdpchandler_SDK import *
import pandas as pd
import time
import datetime

# Names of how to store CSV files containing sensor data
logName = "test"
logMap = "dataCSV"

# Dataframe containing sensor data
df = pd.DataFrame(columns=['Device', 'EulerX', 'EulerY', 'EulerZ', 'FreeAccX', 'FreeAccY', 'FreeAccZ', 'Timestamp'])

def initialize():
    """
    Initializes the Movella DOTS for data collection.
    """
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
    """
    Sets up the connected Movella DOTS for data acquisition.
    """
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
    """
    Initiates real-time streaming of sensor data
    """
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
                for device in xdpcHandler.connectedDots():
                    packet = xdpcHandler.getNextPacket(device.portInfo().bluetoothAddress())
                    if packet.containsOrientation() and packet.containsFreeAcceleration():
                        euler = packet.orientationEuler()
                        freeAcc = packet.freeAcceleration()

                        print(
                            f"Device: {device.bluetoothAddress()}, EulerX:{euler.x():7.2f}, EulerY:{euler.y():7.2f}, EulerZ:{euler.z():7.2f}, freeAcc:{freeAcc}"
                        )

                        # Append data to dataframe
                        df.loc[len(df)] = [device.bluetoothAddress(),
                                           euler.x(), euler.y(), euler.z(),
                                           freeAcc[0], freeAcc[1], freeAcc[2],
                                           timestamp]



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
    """
    Converting dataframe to CSV file with specified name and directory
    """
    print("Saving dataframe to CSV")

    # Define the file path
    file_path = f"{mapLog}/{nameLog}.csv"

    # Save the DataFrame to CSV
    if not df.to_csv(file_path, index=False):
        print("Could not save dataframe to CSV")

    print(f"DataFrame saved to {file_path}")

if __name__ == "__main__":
    # Initialize the XdpcHandler and start time
    xdpcHandler = XdpcHandler()
    start_time = time.time()

    # Initialize and set up Movella DOTS
    initialize()
    set_up()
    print("Press ENTER to redo initialization and setup or press SPACE to start measuring.")

    # Listener for key events to start measurement or redo initialization
    def on_press(key):
        if key == keyboard.Key.enter:
            initialize()
            set_up()
            print("Press ENTER to redo initialization and setup or press SPACE to start measuring.")
        elif key == keyboard.Key.space:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # Start measurement
    measurement()

    # Sto measurement
    stopMeasurement()

    # Reset time to name CSV file
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df_to_csv(mapLog=logMap, nameLog=f"{logName}_{current_time}")
    print("Press ENTER to stop measurements or press SPACE to do another measurement")

    def on_press(key):
        if key == keyboard.Key.space:
            # Reset DataFrame for another stream
            df.drop(df.index, inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Start another measurement
            measurement()

            # Stop measurement
            stopMeasurement()

            # Reset time to name CSV file
            current_time2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            df_to_csv(mapLog=logMap, nameLog=f"{logName}_{current_time2}")
            print("Press ENTER to stop measurements or press SPACE to do another measurement")
        elif key == keyboard.Key.enter:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # Clean up resources
    xdpcHandler.cleanup()


