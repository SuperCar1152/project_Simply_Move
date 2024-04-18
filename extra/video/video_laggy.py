import cv2
from ffpyplayer.player import MediaPlayer

file = "simply dance.mp4"

# Open the video file
video = cv2.VideoCapture(file)

# Initialize the audio player
player = MediaPlayer(file)

while True:
    ret, frame = video.read()
    audio_frame, val = player.get_frame()

    if not ret:
        print("End of video")
        break

    if val != 'eof' and audio_frame is not None:
        # Extract the audio frame from the tuple
        audio_time = audio_frame[0] / 1000.0  # Convert audio time to seconds
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_delay = int((1 / fps) * 1000)  # Convert frame rate to milliseconds

        # Display the frame
        cv2.imshow("Video", frame)

        # Wait for the appropriate delay
        cv2.waitKey(frame_delay)

    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
