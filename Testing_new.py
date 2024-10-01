# Importing libraries
import os
import cv2
import numpy as np
from datetime import datetime

# Custom logging function
def custom_log(message, log_file='custom_object_detection.log'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as file:
        file.write(f'{timestamp} - {message}\n')

# Preprocess the images
# To grayscale then blur the image -> Return the blured frame
def preprocess_image(frame): # Take in a frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Modify frame from color to grayscale
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0) # Blur the frame, this is to help objects stand out
    return blurred_frame # Give back the modified frame

# Detect the laser light
# Take the frame and make only the lazer grid stand out
def detect_laser_light(frame): # Take in a frame
    _, thresholded_frame = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY) # Modify pixel values to make laser light stand out
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Only find the internal contours
    laser_light = max(contours, key=cv2.contourArea) # Assumes the lazer light will be the most white region of pixels
    return laser_light # Return that detected lazer contour 

# Track the position of the laser light
def track_laser_light(laser_light):
    M = cv2.moments(laser_light) # Calc centroid of the laser light
    cX = int(M["m10"] / M["m00"]) # Calc centroid of the laser light X
    cY = int(M["m01"] / M["m00"]) # Calc centroid of the laser light Y
    return (cX, cY) # Return the centroid of the laser light X,Y

# Calculate the deflection
def calculate_deflection(positions):
    if len(positions) >= 2:  # Ensure there are at least two positions to calculate deflection
        deflection = (positions[-1][0] - positions[0][0], positions[-1][1] - positions[0][1]) # Calculate the deflection by subtracting the first position from the last position
    else:
        deflection = (0, 0)  # Default deflection if there are not enough positions
    return deflection

# Main loop
def main():
    cap = cv2.VideoCapture('/content/LazerGrid_ThickLine.mp4')  # Use the video file as input
    positions = []
    frame_counter = 0
    desired_width, desired_height = 640, 480  # Set your desired width and height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the desired format
        frame = cv2.resize(frame, (desired_width, desired_height))

        frame_counter += 1
        custom_log(f'Start processing frame {frame_counter}')

        frame = preprocess_image(frame)
        custom_log('Frame preprocessed')

        laser_light = detect_laser_light(frame)
        custom_log('Laser light detected')

        position = track_laser_light(laser_light)
        positions.append(position)
        custom_log(f'Laser light position: {position}')

        # Draw the laser light position on the frame
        cv2.circle(frame, position, 5, (0, 255, 0), -1)
        cv2.putText(frame, f'Position: {position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Result frame: ', frame)

        # Exit the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    deflection = calculate_deflection(positions)
    custom_log(f'Deflection: {deflection}')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()