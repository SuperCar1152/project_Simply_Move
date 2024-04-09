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


logName = "V1"
logMap = "dataCSV"


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
    # elif len(xdpcHandler.connectedDots()) < 5:
    #     print("Not connected to 5 DOTs, trying again.")

    print(f"Connected number of DOTs: {len(xdpcHandler.connectedDots())}")
    print(xdpcHandler.connectedDots())

def set_up():
    """
    Set up each device from the connected devices
    - Filter Profiles
        General: Default setting, meaning moderate dynamics with homogeneous dynamic field.
        Dynamic: Setting for fast and jerky motions that last for a short time.
                 Uses magnetometer for stabilization. (Used for example for sprinting)
    - Ouput Rate:
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
        if not device.setOutputRate(60):
            print("Setting output rate failed!")

        # Set logging options
        device.setLogOptions(movelladot_pc_sdk.movelladot_pc_sdk_py310_64.XsLogOptions_Euler)
        # Enable logging
        logFileName = f"{logMap}\logfile_{device.bluetoothAddress().replace(':', '-')}_{logName}.csv"
        if not device.enableLogging(logFileName):
            print(f"Failed to enable logging. Reason: {device.lastResultText()}")

        # Start measurement mode
        if not device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler):
            print(f"Could not put device into measurement mode. Reason: {device.lastResultText()}")

        print(f"Succesfully setup {device}")

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
    print(f"Succesfully synced {len(deviceList)}")
    return manager

def measurement():
    print(f"Starting measurement for 90 seconds")
    startTime = movelladot_pc_sdk.XsTimeStamp_nowMs()
    # Continue acquiring data for a certain duration (e.g., 90 seconds)
    while movelladot_pc_sdk.XsTimeStamp_nowMs() - startTime <= 90000:
        # Check if packets are available for processing from connected devices
        if xdpcHandler.packetsAvailable():
            s = ""
            # For every connected device
            for device in xdpcHandler.connectedDots():
                # Retrieves the next available packet, and stores it in the packet variable.
                packet = xdpcHandler.getNextPacket(device.portInfo().bluetoothAddress())
                # Check if packet contains Orientation data, extracts Euler angles.
                if packet.containsOrientation():
                    euler = packet.orientationEuler()
                    s += (
                        f"startTime: {startTime}, TS: {packet.sampleTimeFine():8d}, EulerX:{euler.x():7.2f}, EulerY:{euler.y():7.2f}, EulerZ:{euler.z():7.2f}| ")
            # Print the Euler Angles
            print("%s\r" % s, end="", flush=True)

    print("Finished Measuring")

def stop(manager):
    print("Stopping Measurement")
    # Stop measurement and synchronization
    for device in xdpcHandler.connectedDots():
        if not device.stopMeasurement():
            print("Failed to stop measurement.")
        if not device.disableLogging():
            print("Failed to disable logging.")
    if not manager.stopSync():
        print("Failed to stop syncing")

    # Cleanup
    xdpcHandler.cleanup()
    print("Data acquisition complete, stopped measurement")

if __name__ == "__main__":
    xdpcHandler = XdpcHandler()
    initialize()
    set_up()
    manager = sync()
    measurement()
    stop(manager)


