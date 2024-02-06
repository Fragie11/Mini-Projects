#Tejas Agrawal 
# Task 2 for the AI AI Engineer - Computer Vision & NLP 

import cv2
import numpy as np

# Define HSV range for green color
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Specify the input and output video paths
input_video_path = "D:/python/task_2_video.mp4"
output_video_path = "D:/python/task_2_output_video.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask using HSV range for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a red dot at the center of each significant green polka dot
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust the area threshold based on your specific video
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw a red dot at the center
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Output Video", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
