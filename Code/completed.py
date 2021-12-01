import cv2
import numpy as np
import imutils
from keras.preprocessing.image import img_to_array
from keras.models import load_model

cascade_file_address = '../lbp/cascade.xml'
weight_file_address = './models/_mini_XCEPTION.75-0.64.hdf5'

face_detection = cv2.CascadeClassifier(cascade_file_address)
expressions_classifier = load_model(weight_file_address, compile=False)
Expressions = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

cv2.namedWindow('stream')
cap = cv2.VideoCapture(0)
while True:
    error,image = cap.read()
    image = imutils.resize(image,width=480)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    stochastics_page = np.zeros((400, 400, 3),dtype="uint8")
    image2 = image.copy()
    if len(faces) >= 1:
        (X, Y, WIDTH, HEIGHT) = sorted(faces, reverse=True,key=lambda x: x[2]*x[3])[0]
        ROI = gray[Y:Y + HEIGHT, X:X + WIDTH]
        ROI = cv2.resize(ROI, (48, 48))
        ROI = ROI.astype("float") / 255.0
        ROI = img_to_array(ROI)
        ROI = np.expand_dims(ROI, axis=0)
        predictions = expressions_classifier.predict(ROI)[0]
        label = Expressions[predictions.argmax()]
        for (i, (expression, probability)) in enumerate(zip(Expressions, predictions)):
                text = "{} : {:.2f} %".format(expression, probability * 100)
                w = int(probability * 350)
                cv2.rectangle(stochastics_page, (7, (i * 35) + 5),(w, (i * 35) + 35), (255, 255, 10), -1)
                cv2.putText(stochastics_page, text, (10, (i * 35) + 23),cv2.FONT_HERSHEY_SIMPLEX, 0.55,(255, 255, 255), 1)
                cv2.putText(image2, label, (X+int(0.25*WIDTH), Y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                cv2.rectangle(image2, (X, Y), (X + WIDTH, Y + HEIGHT), (255, 0,0), 2)

    cv2.imshow('stream', image2)
    cv2.imshow("Stochastics", stochastics_page)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
