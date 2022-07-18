import cv2
from random import randrange

# Import Video file
video_file = 'dataset_video1.avi'
video = cv2.VideoCapture(video_file)

# Get the pre-trained car classifier
classifier_file = 'car_detector.xml'

# create a car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

count = 0

while True:

    # Read a frame from the video
    success, frame = video.read()

    # if frame is successfully received
    if success:
        # convert the frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars
        cars = car_tracker.detectMultiScale(grayscale_frame)

        # Draw rectangles around the cars
        for x, y, w, h in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(100,
                          256), randrange(100, 256), randrange(100, 256)), 3)

        # Show the frame with rectangle
        cv2.imshow('Clever program Car Detector', frame)

        key = cv2.waitKey(1)

        if key == 81 or key == 113:
            break
    else:
        if count == 10:
            break
        count += 1

video.release()

print("Code Completed!!")
