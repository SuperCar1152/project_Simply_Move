from xdpchandler import *

if __name__ == "__main__":
    xdpcHandler = XdpcHandler()

    xdpcHandler.initialize()

    xdpcHandler.scanForDots()
    if len(xdpcHandler.detectedDots()) == 0:
        print("No Movella DOT device(s) found. Aborting.")

    xdpcHandler.connectDots()

    if len(xdpcHandler.connectedDots()) == 0:
        print("Could not connect to any Movella DOT device(s). Aborting.")

    print(f"Number of connected DOTs: {len(xdpcHandler.connectedDots())}, press SPACE to try again, ENTER to check battery")
    def on_press(key):
        if key == keyboard.Key.space:
            xdpcHandler.initialize()
            xdpcHandler.scanForDots()
            if len(xdpcHandler.detectedDots()) == 0:
                print("No Movella DOT device(s) found.")
            xdpcHandler.connectDots()
            if len(xdpcHandler.connectedDots()) == 0:
                print("Could not connect to any Movella DOT device(s).")

            print(
                f"Number of connected DOTs: {len(xdpcHandler.connectedDots())}, press SPACE to try again, ENTER to check battery")
        elif key == keyboard.Key.enter:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    for device in xdpcHandler.connectedDots():
        print(f"Device: {device.bluetoothAddress()}, has battery: {device.batteryLevel()}")

    xdpcHandler.cleanup()

