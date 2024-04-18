import cv2
from moviepy.editor import VideoFileClip

# Load the video
video_path = 'simply dance.mp4'
clip = VideoFileClip(video_path)

# Get the video's frame rate
fps = clip.fps

# Create a video capture object
cap = cv2.VideoCapture(video_path)

# Play the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Display the frame
    cv2.imshow('Video', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
