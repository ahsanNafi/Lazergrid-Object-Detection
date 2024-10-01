""" 
Aerium Real Time Data Collection

This file reads data from a camera, processes it and then uses a detection algorithm
to detect the presence of an anomoly. If an anomoly is detected, the program will log that to
an output file.

"""

# Importing libraries
import os
import cv2
import numpy as np
from datetime import datetime
import time
import sys


def get_video_feed(timeout=60):
    """Detect and choose the correct video capture device."""
    start_time = time.time()
    attempts = 0
    while time.time() - start_time < timeout:
        devices = [device for device in os.listdir('/dev') if device.startswith('video0')]
        attempts += 1
        if devices:
            print(f"Attempt {attempts}: Found video device(s) after {time.time() - start_time:.1f} seconds")
            return 0
        print(f"Attempt {attempts}: No video devices found. Retrying in 5 seconds...")
        time.sleep(5)  # Check every 5 seconds
    print(f"No video devices found after {timeout} seconds. Exiting...")
    sys.exit(1)



def custom_log(message, log_file='custom_object_detection.log'):
    """Custom logging function."""
    """Creates and logs to a log file, formats message and datetime"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as file:
        file.write(f'{timestamp} - {message}\n')



def save_video(frames, fps, video_format, width, height):
    """Function to save the processed frames as a video file"""

    # Get the current timestamp
    timestamp = datetime.now().strftime('%y-%m-%d %H:%M:%S')

    # Define the filename with the timestamp
    filename = f'output_{timestamp}.mp4'

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Write the frames to the output video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

def detect_and_identify_objects(frame, frame_counter):
    """Function to detect red laser light and identify objects within it"""

    custom_log(f'Start processing frame {frame_counter}')

    # Define the lower and upper bounds for the red laser light color in HSV
    lower_red = np.array([130, 100, 100])  # Lower bound for red
    upper_red = np.array([270, 255, 255])  # Upper bound for red

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the red laser light color
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an object counter
    object_counter = 0

    # Draw contours around red laser light and label objects
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Set a minimum threshold for contour area to filter out small noise
        if area > 50:
            object_counter += 1
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, f'Object {object_counter}', (contour[0][0][0], contour[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Log details of each detected object
            custom_log(f'Frame {frame_counter}: Detected Object {object_counter} - Area: {area}, Position: {contour[0][0]}')

    # Log the total number of objects detected in the frame
    custom_log(f'Frame {frame_counter}: Total detected objects: {object_counter}')

    return frame


def main():
    """Main function to read video feed, process frames and detect objects"""

    cap = cv2.VideoCapture(get_video_feed())

    #change the desired width and height of the frame to 1280x720 for better performance
    desired_width = 1920
    desired_height = 1080
    frame_counter = 0


    #variable to save the processed frames
    fps = 23.0
    video_format = 'mp4v'
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (desired_width, desired_height))

        result_frame = detect_and_identify_objects(frame, frame_counter)
        processed_frames.append(result_frame)

        cv2.imshow("Result frames ", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

    custom_log('------------------Finished processing all frames------------------')
    save_video(processed_frames, fps, video_format, desired_width, desired_height)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()