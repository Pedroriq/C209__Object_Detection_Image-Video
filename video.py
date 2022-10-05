import cv2  
import numpy as np
import time

#importando o vídeo
cap = cv2.VideoCapture('video.mp4')


if (cap.isOpened()== False):
  print("Error opening video stream or file")

#classifier dos carros
car_cascade = cv2.CascadeClassifier('cars.xml')

while(cap.isOpened()):

    #ret - frame disponivel/frames - proximo frame (array)
    ret, frames = cap.read()
    
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
      
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
      
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
  
   # Display frames in a window 
    cv2.imshow('video2', frames)
      
    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break


cv2.destroyAllWindows()