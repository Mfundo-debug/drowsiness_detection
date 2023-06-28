import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()
sound = mixer.Sound('C:/Users/didit/Downloads/open-cv/drowsiness_detection/Drowsiness detection/alarm.wav')

face = cv2.CascadeClassifier('C:/Users/didit/Downloads/open-cv/drowsiness_detection/Drowsiness detection/haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('C:/Users/didit/Downloads/open-cv/drowsiness_detection/Drowsiness detection/haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('C:/Users/didit/Downloads/open-cv/drowsiness_detection/Drowsiness detection/haar cascade files/haarcascade_righteye_2splits.xml')

labels = ['Closed', 'Open']

model = load_model('C:/Users/didit/Downloads/open-cv/drowsiness_detection/Drowsiness detection/models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=6, scaleFactor=1.2, minSize=(20, 20))

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    eyes_closed = True

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

        left_eye = leye.detectMultiScale(gray[y:y+h, x:x+w])
        right_eye = reye.detectMultiScale(gray[y:y+h, x:x+w])

        for (ex, ey, ew, eh) in right_eye:
            r_eye = gray[y+ey:y+ey+eh, x+ex:x+ex+ew]
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = np.expand_dims(r_eye, axis=0)
            r_eye = np.reshape(r_eye, (24, 24, -1))
            rpred = model.predict(np.array([r_eye]))
            if np.any(rpred == 1):
                label = 'Open'
            if np.all(rpred == 0):
                label = 'Closed'
                eyes_closed = True
            else:
                eyes_closed = False
            break

        for (ex, ey, ew, eh) in left_eye:
            l_eye = gray[y+ey:y+ey+eh, x+ex:x+ex+ew]
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = np.expand_dims(l_eye, axis=0)
            l_eye = np.reshape(l_eye, (24, 24, -1))
            lpred = model.predict(np.array([l_eye]))
            if np.any(lpred == 1):
                label = 'Open'
            if np.all(lpred == 0):
                label = 'Closed'
                eyes_closed = True
            else:
                eyes_closed = False
            break

        if eyes_closed:
            score += 1
            cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0

        cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score > 5:
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()
            except:
                pass
        else:
            sound.stop()

        if thicc < 4:
            thicc += 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 