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
import pandas as pd
import time
from tensorflow.keras import models
import numpy as np
import sys
import pygame
import random

df = pd.DataFrame(columns=['Device', 'EulerX', 'EulerY', 'EulerZ', 'FreeAccX', 'FreeAccY', 'FreeAccZ', 'Timestamp'])

Moves = ['Clap', 'ArmsOut', 'Elbow', 'Jump', 'Guitar']

difficulty = 3

movement_counts = {move: 0 for move in Moves}

model = models.load_model('model.keras')

def initialize():
    print("Initializing DOTs")

    # Initialize
    if not xdpcHandler.initialize():
        print("Failed to initialize SDK. Closing program.")
        xdpcHandler.cleanup()

    # Scan for XSense DOTS
    xdpcHandler.scanForDots()
    if len(xdpcHandler.detectedDots()) == 0:
        print("No Movella DOT device(s) found. Closing program.")
        xdpcHandler.cleanup()

    # Connect to detected DOTS
    xdpcHandler.connectDots()
    if len(xdpcHandler.connectedDots()) == 0:
        print("Could not connect to any Movella DOT device(s). Closing program.")
        xdpcHandler.cleanup()

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
        print(predicted_class)

        # Check if the timestamp falls within each time interval and count the corresponding movement
        # Calculate elapsed time within each interval
        clock = int(timestamp - start_time)
        print(clock)
        if clock <= clap1_interval + difficulty and predicted_class == 'Clap':
            movement_counts['Clap'] += 1
        elif clap1_interval - difficulty < clock <= guitar1_interval + difficulty and predicted_class == 'Guitar':
            movement_counts['Guitar'] += 1
        elif guitar1_interval - difficulty < clock <= jump1_interval + difficulty and predicted_class == 'Jump':
            movement_counts['Jump'] += 1
        elif jump1_interval - difficulty < clock <= elbow1_interval + difficulty and predicted_class == 'Elbow':
            movement_counts['Elbow'] += 1
        elif elbow1_interval - difficulty < clock <= armsOut1_interval + difficulty and predicted_class == 'ArmsOut':
            movement_counts['ArmsOut'] += 1
        elif armsOut1_interval - difficulty < clock <= jump2_interval + difficulty and predicted_class == 'Jump':
            movement_counts['Jump'] += 1
        elif jump2_interval - difficulty < clock <= elbow2_interval + difficulty and predicted_class == 'Elbow':
            movement_counts['Elbow'] += 1
        elif elbow2_interval - difficulty < clock <= armsOut2_interval + difficulty and predicted_class == 'ArmsOut':
            movement_counts['ArmsOut'] += 1
        elif clock >= armsOut2_interval + difficulty:
            print("Done")
        else:
            print("Not doing the correct move")

def stopStream():
    """
    Stopping all measurements on all devices
    """
    print("Stopping Measurement mode")
    for device in xdpcHandler.connectedDots():
        if not device.stopMeasurement():
            print("Failed to stop measurement mode.")
    print("Stream completed")

def show_end_screen(scores):
    pygame.init()

    # Set up the screen
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("End Screen")

    # Set up fonts
    font_title = pygame.font.Font("end_screen_visuals/Milker.otf", 72)  # Replace "Milker.otf" with the path to your desired font file
    font_scores = pygame.font.Font("end_screen_visuals/Milker.otf", 36)  # Replace "Milker.otf" with the path to your font file

    # Colors
    background_color = (0, 0, 0)  # Black background

    # Load dance images and resize them
    dance_images = [
        pygame.transform.scale(pygame.image.load("end_screen_visuals/Dance1.png"), (200, 300)),
        pygame.transform.scale(pygame.image.load("end_screen_visuals/Dance2.png"), (200, 300)),
        pygame.transform.scale(pygame.image.load("end_screen_visuals/Dance3.png"), (200, 300)),
    ]
    dance_index = 0
    dance_image = dance_images[dance_index]

    # Animation settings
    animation_delay = 0.05

    # Title animation settings
    title_colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128)]  # Red, Orange, Yellow, Green, Blue, Purple
    title_color_index = 0
    title_rotation_angle = 90

    # Score display settings
    score_delay = 500  # Delay between displaying each score in milliseconds
    last_score_time = pygame.time.get_ticks()
    displayed_scores = []  # List to keep track of displayed scores

    # Main loop
    running = True
    score_keys = list(scores.keys())
    score_index = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill(background_color)

        # Display dancing animation
        dance_rect = dance_image.get_rect(bottomleft=(20, screen_height - 20))  # Adjust position to bottom-left corner
        screen.blit(dance_image, dance_rect)

        # Update dance animation
        if random.randint(1, 100) < 5:  # Randomly change dance animation
            dance_index = (dance_index + 1) % len(dance_images)
            dance_image = dance_images[dance_index]

        # Display title text with changing colors
        title_color_index = (title_color_index + 1) % len(title_colors)
        title_color = title_colors[title_color_index]
        title_text = pygame.transform.rotate(font_title.render("Simply Move", True, title_color), title_rotation_angle)
        title_text_rect = title_text.get_rect(midright=(screen_width - 50, screen_height // 2))  # Adjust position to midright
        screen.blit(title_text, title_text_rect)

        # Display scores with delay
        current_time = pygame.time.get_ticks()
        if current_time - last_score_time >= score_delay and score_index < len(score_keys):
            player = score_keys[score_index]
            score = scores[player]
            displayed_scores.append((player, score))  # Add the score to the displayed scores list
            score_index += 1
            last_score_time = current_time

        # Render and display all displayed scores
        y = 50
        for displayed_score in displayed_scores:
            player, score = displayed_score
            text = font_scores.render(f"{player}: {score}", True, (255, 255, 255))  # White text
            text_rect = text.get_rect(center=(screen_width // 2, y))
            screen.blit(text, text_rect)
            y += 50

        pygame.display.flip()

        # Add animation delay
        pygame.time.delay(int(animation_delay * 1000))
    pygame.quit()


if __name__ == "__main__":
    xdpcHandler = XdpcHandler()
    start_time = time.time()
    initialize()
    set_up()
    print("Start screen")
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
    scores = {"Clap count:": movement_counts['Clap'],
              "Jump count:": movement_counts['Jump'],
              "Guitar count:": movement_counts['Guitar'],
              "Elbow count:": movement_counts['Elbow'],
              "ArmsOut count:": movement_counts['ArmsOut'],
              "Total score:": movement_counts['Clap'] +
                              movement_counts['Jump'] + movement_counts['Guitar'] +
                              movement_counts['Elbow'] + movement_counts['ArmsOut']
              }
    show_end_screen(scores)
    stopStream()
    print("Press ENTER to stop ALL streams or press SPACE to do another stream")

    def on_press(key):
        if key == keyboard.Key.space:
            movement_counts['Clap'] = 0
            movement_counts['Jump'] = 0
            movement_counts['Elbow'] = 0
            movement_counts['ArmsOut'] = 0
            movement_counts['Guitar'] = 0

            df.drop(df.index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            stream()
            scores2 = {"Clap count:": movement_counts['Clap'],
                      "Jump count:": movement_counts['Jump'],
                      "Guitar count:": movement_counts['Guitar'],
                      "Elbow count:": movement_counts['Elbow'],
                      "ArmsOut count:": movement_counts['ArmsOut'],
                      "Total score:": movement_counts['Clap'] +
                                      movement_counts['Jump'] + movement_counts['Guitar'] +
                                      movement_counts['Elbow'] + movement_counts['ArmsOut']
                      }
            show_end_screen(scores2)
            stopStream()
            print("Press ENTER to stop ALL streams or press SPACE to do another stream")
        elif key == keyboard.Key.enter:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    xdpcHandler.cleanup()
    sys.exit()


