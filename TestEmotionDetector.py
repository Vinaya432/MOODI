import cv2
import numpy as np
from keras.models import model_from_json
import time
import random
import os

emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}
text = ""
# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

now = time.time()
future = now + 5

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        text = emotion_dict[maxindex]
        text = text.title()
    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(30) & 0xFF

    if time.time() > future:

        cv2.destroyAllWindows()

    #emotion and song mapping
        if text == 'Angry':
                print('You are angry !!!! please calm down:) ,I will play song for you :')
                path = "songs\\angry\\"
                files = os.listdir(path)
                d = random.choice(files)
                os.startfile(path + d)
                break

        if text == 'Happy':
                print('You are smiling :) ,I playing special song for you: ')
                path = "songs\\happy\\"
                files = os.listdir(path)
                d = random.choice(files)
                os.startfile(path + d)
                break

        if text == 'Neutral':
                print('Yo cheer up ,I will play a song for you: ')
                path = "songs\\neutral\\"
                files = os.listdir(path)
                d = random.choice(files)
                os.startfile(path+d)
                break

        if text == 'Sad':
                print('You are sad,dont worry:) ,I playing song for you: ')
                path = "songs\\sad\\"
                files = os.listdir(path)
                d = random.choice(files)
                os.startfile(path + d)
                break




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

