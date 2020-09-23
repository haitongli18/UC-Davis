from collections import deque
import csv
import cv2
import time

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
pts = deque([])
pts_size = 20
# file = open('test_movement.csv', 'a')
# writer = csv.writer(file)

while(True):
    begin_time = time.time()

    ret, frame = cap.read()
    # we don't need color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resize so that things are faster
    resized = cv2.resize(gray, (16 * 30, 9 * 30))
    # detect faces
    faces = face_cascade.detectMultiScale(resized, 1.3, 5)
    # draw the rectangle, dots and lines
    for (x,y,w,h) in faces:
        resized = cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)

        center_x = int(x + w/2)
        center_y = int(y + h/2)
        # writer.writerow([center_x, center_y])

        if len(pts) == pts_size:
            pts.pop()
            pts.appendleft((center_x, center_y))
        else:
            pts.appendleft((center_x, center_y))

        resized = cv2.circle(resized, (center_x, center_y), 0, (255, 255, 255), 5)
        for i in range(len(pts)):
            if i == len(pts) - 1:
                continue
            resized = cv2.line(resized, pts[i], pts[i+1], (0, 0, 0), 1)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = resized[y:y+h, x:x+w]
    cv2.imshow('frame', resized)

    end_time = time.time()
    print('loop costs {} seconds.'.format(end_time - begin_time))
    if cv2.waitKey(24) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# file.close()