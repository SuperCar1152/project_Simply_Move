"""
 Copyright (c) 2003-2023 Movella Technologies B.V. or subsidiaries worldwide.
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 1.	Redistributions of source code must retain the above copyright notice,
 	this list of conditions and the following disclaimer.

 2.	Redistributions in binary form must reproduce the above copyright notice,
 	this list of conditions and the following disclaimer in the documentation
 	and/or other materials provided with the distribution.

 3.	Neither the names of the copyright holders nor the names of their contributors
 	may be used to endorse or promote products derived from this software without
 	specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR
 TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Requires installation of the correct Movella DOT PC SDK wheel through pip
For example, for Python 3.9 on Windows 64 bit run the following command
pip install movelladot_pc_sdk-202x.x.x-cp39-none-win_amd64.whl
"""

import movelladot_pc_sdk.movelladot_pc_sdk_py310_64

from xdpchandler_SDK import *
xdpcHandler = XdpcHandler()

version = 1

def main():
    """
    Initialize
    """
    # If XdpcHandler is not initialized -> Cleanup
    if not xdpcHandler.initialize():
        xdpcHandler.cleanup()
        exit(-1)

    """
    Scan
    """
    xdpcHandler.scanForDots()
    # If no XSense Dots are detected -> Cleanup
    if len(xdpcHandler.detectedDots()) == 0:
        print("No Movella DOT device(s) found. Aborting.")
        xdpcHandler.cleanup()
        exit(-1)

    """
    Connect
    """
    # Connect to XSense Dots (Create connection)
    xdpcHandler.connectDots()
    # If no XSense Dots are connected -> Cleanup
    if len(xdpcHandler.connectedDots()) == 0:
        print("Could not connect to any Movella DOT device(s). Aborting.")
        xdpcHandler.cleanup()
        exit(-1)

    # For every connected XSense Dot (device)
    for device in xdpcHandler.connectedDots():
        """
        Filter Profiles
            - General: Default setting, meaning moderate dynamics with homogeneous dynamic field.
            - Dynamic: Setting for fast and jerky motions that last for a short time. 
                Uses magnetometer for stabilization. (Used for example for sprinting)
        """
        # Set filter profile to General
        if device.setOnboardFilterProfile("General"):
            print("Successfully set profile to General")
        else:
            print("Setting filter profile failed!")

        # Set output rate to 60 Hz
        if device.setOutputRate(60):
            print("Successfully set output rate to 60 Hz")
        else:
            print("Setting output rate failed!")

        """
        Settings for CSV Output
        """
        # Set output to CSV with Quaternion and Euler Angles
        print("Setting Quaternion and Euler CSV output")
        device.setLogOptions(movelladot_pc_sdk.movelladot_pc_sdk_py310_64.XsLogOptions_QuaternionAndEuler)
        # Setting name for CSV Output
        logFileName = f"dataCSV\logfile_{device.bluetoothAddress().replace(':', '-')}_V{version}.csv"
        print(f"Enable logging to: {logFileName}")
        # Attempts to enable logging to the specified file for the device and prints the outcome if it fails.
        if not device.enableLogging(logFileName):
            print(f"Failed to enable logging. Reason: {device.lastResultText()}")

        """
        Measurement Mode
        """
        # Set device to measurement mode
        print("Putting device into measurement mode.")
        if not device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler):
            print(f"Could not put device into measurement mode. Reason: {device.lastResultText()}")
            continue

    """
    Recording Data
    """
    # Prints message indicating start of main loop
    print("\nMain loop. Recording data for 90 seconds.")
    print("-----------------------------------------")

    # First printing some headers (bluetooth adress), so we see which data belongs to which device
    s = ""
    for device in xdpcHandler.connectedDots():
        s += f"{device.bluetoothAddress():27}"
    print("%s" % s, flush=True)

    # Boolean for Orientation Reset
    # orientationResetDone = False
    # startTime to set measurement time
    startTime = movelladot_pc_sdk.XsTimeStamp_nowMs()

    # Ensure that the loop runs for 90 seconds (90.000 milliseconds).
    # Checks if the current time minus the start time is less than or equal to 90,000 milliseconds (90 seconds).
    while movelladot_pc_sdk.XsTimeStamp_nowMs() - startTime <= 90000:
        # Check if packets are available for processing from connected devices
        if xdpcHandler.packetsAvailable():

            s = ""
            """"
            Receive Euler Angles as Packet
            """
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

    print("\n-----------------------------------------", end="", flush=True)

    """
    Stop measurement
    """
    print("\nStopping measurement...")
    for device in xdpcHandler.connectedDots():
        if not device.stopMeasurement():
            print("Failed to stop measurement.")
        if not device.disableLogging():
            print("Failed to disable logging.")

    xdpcHandler.cleanup()
    print("Successful exit.")


if __name__ == "__main__":
    main()
