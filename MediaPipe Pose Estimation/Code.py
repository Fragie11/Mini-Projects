#This iss the code for MediaPipe Pose Estimation by tejas agrawal :)

import os
import cv2
import mediapipe as mp
import tensorflow as tf

# Set environment variables
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XNNPACK_DISABLED"] = "1"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Specify the input video path
input_video_path = "D:/python/task_4_video.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Draw the skeleton on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow("MediaPipe Pose Estimation", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
