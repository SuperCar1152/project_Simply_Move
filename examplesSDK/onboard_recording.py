from xdpchandler import *


if __name__ == "__main__":
    xdpcHandler = XdpcHandler()

    if not xdpcHandler.initialize():
        xdpcHandler.cleanup()
        exit(-1)

    xdpcHandler.scanForDots()
    if len(xdpcHandler.detectedDots()) == 0:
        print("No Movella DOT device(s) found. Aborting.")
        xdpcHandler.cleanup()
        exit(-1)

    xdpcHandler.connectDots()

    if len(xdpcHandler.connectedDots()) == 0:
        print("Could not connect to any Movella DOT device(s). Aborting.")
        xdpcHandler.cleanup()
        exit(-1)

    for device in xdpcHandler.connectedDots():
        print("Set onboard recording data rate to 60 Hz")
        if device.setOutputRate(60):
            print("Successfully set onboard recording rate")
        else:
            print("Setting onboard recording rate failed!")

        xdpcHandler.resetRecordingStopped()

        print("")
        print("Starting timed onboard recording for 10 seconds.")
        if not device.startTimedRecording(10):
            print(f"Could not start onboard recording. Reason: {device.lastResultText()}")
            continue

        while not xdpcHandler.recordingStopped():
            recordingTimeInfo = device.getRecordingTime()
            ts = movelladot_pc_sdk.XsTimeStamp()
            ts.setMsTime(recordingTimeInfo.startUTC() * 1000)

            s = f"Recording start time: {ts.utcToLocalTime().toXsString()} " \
                f"total time: {recordingTimeInfo.totalRecordingTime()} seconds " \
                f"remaining time: {recordingTimeInfo.remainingRecordingTime()} seconds"
            print("%s\r" % s, end="", flush=True)
            time.sleep(1)

    xdpcHandler.cleanup()
