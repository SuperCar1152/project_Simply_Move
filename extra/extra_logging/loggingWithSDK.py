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

from xdpchandler import *
import cv2

logName = "liveStreamTest"
logMap = "extra"


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
        if not device.setOutputRate(60):
            print("Setting output rate failed!")

        # Set logging options
        device.setLogOptions(movelladot_pc_sdk.movelladot_pc_sdk_py310_64.XsLogOptions_Euler)
        # Enable logging
        logFileName = f"{logMap}\logfile_{device.bluetoothAddress().replace(':', '-')}_{logName}.csv"
        if not device.enableLogging(logFileName):
            print(f"Failed to enable logging. Reason: {device.lastResultText()}")

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
    print(f"Successfully synced {len(deviceList)}")
    return manager


def measurement():
    """
    Start measuring on all devices until 'ENTER' key is pressed
    - Packets
    - Euler etc.
    """
    print("Starting measurement to stop measuring press ENTER")
    for device in xdpcHandler.connectedDots():
        if not device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler):
            print(f"Could not put device into measurement mode. Reason: {device.lastResultText()}")

    def on_press(key):
        if key == keyboard.Key.enter:
            return False
        return True

    with keyboard.Listener(on_press=on_press) as listener:
        while listener.running:
            if xdpcHandler.packetsAvailable():
                # s = ""
                for device in xdpcHandler.connectedDots():
                    packet = xdpcHandler.getNextPacket(device.portInfo().bluetoothAddress())
                    if packet.containsOrientation():
                        euler = packet.orientationEuler()
                        freeAcc = packet.freeAcceleration()
                        timestamp = packet.sampleTimeFine()
                        print(
                            f"Device: {device.bluetoothAddress()}, TS: {timestamp:8d}, EulerX:{euler.x():7.2f}, EulerY:{euler.y():7.2f}, EulerZ:{euler.z():7.2f}, freeAcc:{freeAcc} ")


def stop(manager):
    """
    Stopping all measurements on all devices
    """
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
    print(f"Manager is: f{manager}")
    print("Press ENTER to redo initialization, setup and synchronization or press SPACE to start measuring.")

    def on_press(key):
        if key == keyboard.Key.enter:
            initialize()
            set_up()
            manager = sync()
            print(f"Manager is: f{manager}")
            print("Press ENTER to redo initialization, setup and synchronization or press SPACE to start measuring.")
        elif key == keyboard.Key.space:
            return False  # Stop listener

    # Start listening for key press events
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    measurement()
    stop(manager)
