import cv2
from random import randrange

# Our image
img_file = 'car_image_3.jpeg'

# Our pre-trained car classifier
classifier_file = 'car_detector.xml'

# Create opencv image
img = cv2.imread(img_file)

# Convert image to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars
cars = car_tracker.detectMultiScale(grayscale_img)

# Drawing rectangles around cars
for x, y, w, h in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(100, 256),
                  randrange(100, 256), randrange(100, 256)), 3)

# Show the image
cv2.imshow('Clever program Car Detector', img)

cv2.waitKey()

print("Code Completed!!")
