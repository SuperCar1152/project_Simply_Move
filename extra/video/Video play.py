import cv2
from ffpyplayer.player import MediaPlayer

file = "simply dance sped.mp4"

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

while True:
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

video.release()
cv2.destroyAllWindows()
