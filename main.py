from random import randrange

import cv2

# load some pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascades_frontalface_default.xml')

# choose an image to detect the face in
# img = cv2.imread('./testIMG/ILA_Front.png')

# to capture the video from the webcam
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    # must convert to greyscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # print(face_coordinates)

    # draw rectangles around the face(s)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow("Federico Bertossi's Face Detector", frame)
    key = cv2.waitKey(1)

    # quit when Q is pressed
    if key == 81 or key == 113:
        break

webcam.release()

print("Code Completed!")
