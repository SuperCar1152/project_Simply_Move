

import movelladot_pc_sdk.movelladot_pc_sdk_py310_64

from xdpchandler import *
import pandas as pd
import time
import datetime
from tensorflow.keras import models
import numpy as np

logName = "testStreamVideo"
logMap = "testCSV"

import cv2
from ffpyplayer.player import MediaPlayer

file = "extra/video/simply dance sped.mp4"

# Set the desired frame rate
desired_fps = 50

# Open the video file
video = cv2.VideoCapture(file)

# Set the frame rate of the video capture object
video.set(cv2.CAP_PROP_FPS, desired_fps)

# Initialize the audio player
player = MediaPlayer(file)

# Calculate the delay between frames based on the desired frame rate
delay = int(1000 / desired_fps)

df = pd.DataFrame(columns=['Device', 'EulerX', 'EulerY', 'EulerZ', 'FreeAccX', 'FreeAccY', 'FreeAccZ', 'Timestamp'])

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
                ret, frame = video.read()
                audio_frame, val = player.get_frame()

                if not ret:
                    print("End of video")
                    break

                cv2.imshow("Video", frame)

                if cv2.waitKey(delay) == ord("q"):
                    break

                if val != 'eof' and audio_frame is not None:
                    # Audio
                    img, t = audio_frame

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

    video.release()
    cv2.destroyAllWindows()

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


