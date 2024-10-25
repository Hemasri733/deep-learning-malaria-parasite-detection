from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog

from PIL import ImageTk, Image
from keras.optimizers import Adam
from keras.models import model_from_json

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2
from imutils import paths
import imutils
import matplotlib.pyplot as plt
#{'negative': 0, 'positive': 1}


main = tkinter.Tk()
main.title("Deep Learning for Smartphone-based Malaria Parasite Detection in Thick Blood Smears")
main.geometry("1300x1200")

global filename
global medianFiltered
global cnn_model
global parasites
global wbc

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Images path loaded\n')


def removeNoise():
    global medianFiltered
    index = 0
    for root, dirs, files in os.walk(filename):
        for fdata in files:
            if index == 0:
                rawImage = cv2.imread(root+"/"+fdata,0)
                rawImage = cv2.resize(rawImage,(800,800))
                blur = cv2.GaussianBlur(rawImage,(5,5),0)
                retval, thresholded = cv2.threshold(rawImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                medianFiltered = cv2.medianBlur(thresholded,5)
                contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour_list = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100 :
                        contour_list.append(contour)
                cv2.drawContours(medianFiltered, contour_list,  -1, (255,0,0), 2)
                if len(contour_list) < 15:
                    cv2.imwrite('data/negative/'+fdata,medianFiltered)
                else:
                    cv2.imwrite('data/positive/'+fdata,medianFiltered)
                index = 1    
    cv2.imshow('Image After Removing Noise',medianFiltered)
    cv2.waitKey(0)                
            
    


def CNN():
    global cnn_indices
    global cnn_model
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(output_dim = 128, activation = 'relu'))
    cnn_model.add(Dense(output_dim = 2, activation = 'softmax'))
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
						 shuffle=True)
    test_set = test_datagen.flow_from_directory('data/train',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical',
					    shuffle=False)
    cnn_model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 2000)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            type_model_json = json_file.read()
            cnn_model = model_from_json(type_model_json)

        cnn_model.load_weights("model/model_weights.h5")
        cnn_model._make_predict_function()   
        cnn_model.summary()
        text.delete('1.0', END)
        text.insert(END,"CNN Model Generated. See black console to check CNN Layers\n");
    else:
        cnn_model = Sequential()
        cnn_model.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
        cnn_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(output_dim = 128, activation = 'relu'))
        cnn_model.add(Dense(output_dim = 2, activation = 'softmax'))
        cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        train_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
						 shuffle=True)
        test_set = test_datagen.flow_from_directory('data/train',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical',
					    shuffle=False)
        cnn_model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 2000)
        cnn_model.save_weights('model/model_weights.h5')
        model_json = cnn_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        print(training_set.class_indices)
        print(cnn_model.summary())
        text.delete('1.0', END)
        text.insert(END,"CNN Model Generated. See black console to check CNN Layers\n");



def predictParasite():
     global parasites
     global wbc
     parasites = 0
     wbc = 0
     filename = filedialog.askopenfilename(initialdir="testimage")
     rawImage = cv2.imread(filename,0)
     rawImage = cv2.resize(rawImage,(800,800))
     blur = cv2.GaussianBlur(rawImage,(5,5),0)
     retval, thresholded = cv2.threshold(rawImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     medianFiltered = cv2.medianBlur(thresholded,5)
     contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     contour_list = []
     for contour in contours:
         area = cv2.contourArea(contour)
         if area > 100 :
             contour_list.append(contour)
     cv2.drawContours(medianFiltered, contour_list,  -1, (255,0,0), 2)
     cv2.imwrite('test.png',medianFiltered)
     imagetest = image.load_img('test.png', target_size = (128,128))
     imagetest = image.img_to_array(imagetest)
     imagetest = np.expand_dims(imagetest, axis = 0)
     preds = cnn_model.predict(imagetest)
     print(str(preds)+" "+str(np.argmax(preds)))
     predict = np.argmax(preds)
     msg = ""
     wbc = len(contour_list)
     if predict == 0:
         msg = "Test Result Negative"
         imagedisplay = cv2.imread('test.png')
         orig = imagedisplay.copy()       
         output = imutils.resize(orig, width=400)
         cv2.putText(output, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
         cv2.imshow("Predicted Result", output)
         cv2.waitKey(0)  
     else:
         msg = "Test Result Positive"
         contours_poly = [None]*len(contours)
         boundRect = [None]*len(contours)
         centers = [None]*len(contours)
         radius = [None]*len(contours)
         for i, c in enumerate(contours):
             contours_poly[i] = cv2.approxPolyDP(c, 3, True)
             boundRect[i] = cv2.boundingRect(contours_poly[i])
             centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
         medianFiltered = cv2.cvtColor(medianFiltered,cv2.COLOR_GRAY2RGB)
         for i in range(len(contour_list)):
            if len(contour_list[i]) <= 25:
                parasites = parasites + 1
                cv2.rectangle(medianFiltered, (int(boundRect[i][0]), int(boundRect[i][1])), \
                             (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,0,255), 2)
         cv2.imwrite('test.png',medianFiltered)       
         imagedisplay = cv2.imread('test.png')
         orig = imagedisplay.copy()       
         output = imutils.resize(orig, width=400)
         cv2.putText(output, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
         cv2.imshow("Predicted Result", output)
         cv2.waitKey(0)       

def graph():
    global parasites
    global wbc
    wbc = wbc - parasites
    height = [wbc, parasites]
    bars = ('WBC Count', 'Parasites Count')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
     
    
def videoPredictParasite():
    global parasites
    global wbc
    parasites = 0
    wbc = 0
    videofile = askopenfilename(initialdir = "video")
    video = cv2.VideoCapture(videofile)
    while(True):
        ret, frame = video.read()
        print(ret)
        if ret == True:
             rawImage = frame
             rawImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
             rawImage = cv2.resize(rawImage,(800,800))
             blur = cv2.GaussianBlur(rawImage,(5,5),0)
             retval, thresholded = cv2.threshold(rawImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
             medianFiltered = cv2.medianBlur(thresholded,5)
             contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
             contour_list = []
             for contour in contours:
                 area = cv2.contourArea(contour)
                 if area > 100 :
                     contour_list.append(contour)
             cv2.drawContours(medianFiltered, contour_list,  -1, (255,0,0), 2)
             cv2.imwrite('test.png',medianFiltered)
             imagetest = image.load_img('test.png', target_size = (128,128))
             imagetest = image.img_to_array(imagetest)
             imagetest = np.expand_dims(imagetest, axis = 0)
             preds = cnn_model.predict(imagetest)
             print(str(preds)+" "+str(np.argmax(preds)))
             predict = np.argmax(preds)
             msg = ""
             wbc = len(contour_list)
             if predict == 0:
                 msg = "Test Result Negative"
                 imagedisplay = cv2.imread('test.png')
                 orig = imagedisplay.copy()       
                 output = imutils.resize(orig, width=400)
                 cv2.putText(output, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
                 cv2.imshow("Predicted Result", output)
                   
             else:
                 msg = "Test Result Positive"
                 contours_poly = [None]*len(contours)
                 boundRect = [None]*len(contours)
                 centers = [None]*len(contours)
                 radius = [None]*len(contours)
                 for i, c in enumerate(contours):
                     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                     boundRect[i] = cv2.boundingRect(contours_poly[i])
                     centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
                 medianFiltered = cv2.cvtColor(medianFiltered,cv2.COLOR_GRAY2RGB)
                 for i in range(len(contour_list)):
                     if len(contour_list[i]) <= 25:
                         parasites = parasites + 1
                         cv2.rectangle(medianFiltered, (int(boundRect[i][0]), int(boundRect[i][1])), \
                             (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,0,255), 2)
                 cv2.imwrite('test.png',medianFiltered)       
                 imagedisplay = cv2.imread('test.png')
                 orig = imagedisplay.copy()       
                 output = imutils.resize(orig, width=400)
                 cv2.putText(output, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
                 cv2.imshow("Predicted Result", output)
             if cv2.waitKey(350) & 0xFF == ord('q'):
                break                
        else:
            break
    video.release()
    cv2.destroyAllWindows()


    
font = ('times', 16, 'bold')
title = Label(main, text='Deep Learning for Smartphone-based Malaria Parasite Detection in Thick Blood Smears')
title.config(bg='PaleGreen2', fg='Khaki4')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Thick Blood Images", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

noiseButton = Button(main, text="Remove Noise", command=removeNoise)
noiseButton.place(x=700,y=200)
noiseButton.config(font=font1) 

cnnButton = Button(main, text="Generate CNN Training Model", command=CNN)
cnnButton.place(x=700,y=250)
cnnButton.config(font=font1) 

predictButton = Button(main, text="Upload Test Image & Predict Parasite", command=predictParasite)
predictButton.place(x=700,y=300)
predictButton.config(font=font1)

graphButton = Button(main, text="Extension Predict Parasite From Video", command=videoPredictParasite)
graphButton.place(x=700,y=350)
graphButton.config(font=font1)

exitButton = Button(main, text="Parasite & WBC Count Graph", command=graph)
exitButton.place(x=700,y=400)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='PeachPuff2')
main.mainloop()
