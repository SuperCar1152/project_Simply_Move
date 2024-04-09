import movelladot_pc_sdk.movelladot_pc_sdk_py310_64
from xdpchandler import *

xdpcHandler = XdpcHandler()

def sync_xsense_dots():
    # Initialize SDK
    if not xdpcHandler.initialize():
        print("Failed to initialize SDK")
        return

    # Scan for XSense DOTS
    xdpcHandler.scanForDots()

    # Connect to detected DOTS
    xdpcHandler.connectDots()

    # Set up devices
    for device in xdpcHandler.connectedDots():
        # Set filter profile
        device.setOnboardFilterProfile("General")
        # Set output rate
        device.setOutputRate(60)
        # Set logging options
        device.setLogOptions(movelladot_pc_sdk.movelladot_pc_sdk_py310_64.XsLogOptions_QuaternionAndEuler)
        # Enable logging
        logFileName = "logfile_" + device.bluetoothAddress().replace(':', '-') + ".csv"
        device.enableLogging(logFileName)
        # Start measurement mode
        device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler)

    # Start synchronization among devices
    manager = xdpcHandler.manager()
    deviceList = xdpcHandler.connectedDots()
    startTime = movelladot_pc_sdk.XsTimeStamp_nowMs()
    manager.startSync(deviceList[-1].bluetoothAddress())

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

    # Stop measurement and synchronization
    for device in xdpcHandler.connectedDots():
        device.stopMeasurement()
        device.disableLogging()
    manager.stopSync()

    # Cleanup
    xdpcHandler.cleanup()
    print("Data acquisition complete.")

if __name__ == "__main__":
    sync_xsense_dots()
