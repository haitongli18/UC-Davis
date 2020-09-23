import cv2
import numpy as np
import dlib
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5065

print("UDP target IP: {}".format(UDP_IP))
print("UDP target port: {}".format(UDP_PORT))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    resized = cv2.resize(frame, (16 * 30, 9 * 30))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(resized, (x1, y1), (x2, y2), (255, 255, 255), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if n == 33:
                cv2.circle(resized, (x, y), 2, (0, 0, 255), -1)
                bin_data = bytes("{}, {}".format(x, y), encoding='utf-8')
                sock.sendto(bin_data, (UDP_IP, UDP_PORT))
            else:
                cv2.circle(resized, (x, y), 2, (0, 255, 0), -1)


    cv2.imshow("Frame", resized)

    key = cv2.waitKey(1)
    if key == 27:
        break