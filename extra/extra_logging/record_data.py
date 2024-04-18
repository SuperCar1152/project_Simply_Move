"""
 For this program the SDK of Movella was used.

 Copyright (c) 2003-2023 Movella Technologies B.V. or subsidiaries worldwide.
 All rights reserved.
 Link:

 Adapted by Carlijn le Clercq and Yana Volders
 Data acquisition for XSense Movella DOTs (to CSV Files)

 - Sync and connect till 3 are available
 - Press play to start recording
 - Store in np array?
 """

import movelladot_pc_sdk.movelladot_pc_sdk_py310_64

from xdpchandler_SDK import *
import pandas as pd
import time

logName = "TimeStampFromTimeModule"
logMap = "extra"

df = pd.DataFrame(columns=['Device', 'EulerX', 'EulerY', 'EulerZ', 'FreeAccX', 'FreeAccY', 'FreeAccZ', 'Timestamp'])


def initialize():
    """
    Initialize, scan for and connect to available XSense DOTS (within 20 seconds)
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
    Set up each device from the connected devices
    - Filter Profiles
        General: Default setting, meaning moderate dynamics with homogeneous dynamic field.
        Dynamic: Setting for fast and jerky motions that last for a short time.
                 Uses magnetometer for stabilization. (Used for example for sprinting)
    - Output Rate:
        Explain stuff
    - Logging
        CSV output etc. extendedEuler
        set name etc.
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


def sync():
    """
    Start synchronization for all connected devices
    manager =
    etc.
    """
    # Setup root DOT
    manager = xdpcHandler.manager()
    deviceList = xdpcHandler.connectedDots()
    print(f"Synchronization of {len(deviceList)} devices started")
    if not manager.startSync(deviceList[-1].bluetoothAddress()):
        print(f"Could not start sync. Reason: {manager.lastResultText()}")
    print(f"Successfully synced {len(deviceList)}, with manager: {deviceList[-1].bluetoothAddress()}")
    return manager


def measurement():
    """
    Start measuring on all devices until 'ENTER' key is pressed
    - Packets
    - Euler etc.
    """

    global start_time

    print("Starting measurement to stop measuring press ENTER")
    for device in xdpcHandler.connectedDots():
        if not device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler):
            print(f"Could not put device into measurement mode. Reason: {device.lastResultText()}, trying again")
            if not device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler):
                print(f"Again, could not put device into measurement mode. Reason: {device.lastResultText()}")

    def on_press(key):
        if key == keyboard.Key.enter:
            return False
        return True

    with keyboard.Listener(on_press=on_press) as listener:
        while listener.running:
            if xdpcHandler.packetsAvailable():
                current_time = time.time()
                timestamp = current_time - start_time
                for device in xdpcHandler.connectedDots():
                    packet = xdpcHandler.getNextPacket(device.portInfo().bluetoothAddress())
                    if packet.containsOrientation():
                        euler = packet.orientationEuler()
                        freeAcc = packet.freeAcceleration()

                        print(
                            f"Device: {device.bluetoothAddress()}, EulerX:{euler.x():7.2f}, EulerY:{euler.y():7.2f}, EulerZ:{euler.z():7.2f}, freeAcc:{freeAcc}"
                        )
                        # Append data to dataframe
                        df.loc[len(df)] = [device.bluetoothAddress(),
                                           euler.x(), euler.y(), euler.z(),
                                           freeAcc[0], freeAcc[1], freeAcc[2],
                                           timestamp
                                           ]


def stop(manager):
    """
    Stopping all measurements on all devices
    """
    print("Stopping Measurement")
    # Stop measurement and synchronization
    for device in xdpcHandler.connectedDots():
        if not device.stopMeasurement():
            print("Failed to stop measurement.")
    if not manager.stopSync():
        print("Failed to stop syncing")

    # Cleanup
    xdpcHandler.cleanup()
    print("Data acquisition complete, stopped measurement")

def df_to_csv():
    print("Saving dataframe to CSV")

    # Define the file path
    file_path = f"{logMap}/{logName}.csv"

    # Save the DataFrame to CSV
    if not df.to_csv(file_path, index=False):
        print("Could not save dataframe to CSV")

    print(f"DataFrame saved to {file_path}")

if __name__ == "__main__":
    xdpcHandler = XdpcHandler()
    start_time = time.time()
    initialize()
    set_up()
    manager = sync()

    print("Press ENTER to redo initialization, setup and synchronization or press SPACE to start measuring.")
    def on_press(key):
        if key == keyboard.Key.enter:
            initialize()
            set_up()
            manager = sync()
            print("Press ENTER to redo initialization, setup and synchronization or press SPACE to start measuring.")
        elif key == keyboard.Key.space:
            return False  # Stop listener

    # Start listening for key press events
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    measurement()
    stop(manager)

    df_to_csv()
