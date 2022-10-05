
import cv2 as cv

#Método classifier para a face e os olhos
face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv.CascadeClassifier('haarcascade_eye.xml')

#leitura de uma imagem
image = cv.imread('Imagem1.jpg')
#conversão para escala de cinza
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#detecção das faces 
faces = face_classifier.detectMultiScale(image_gray, 1.3, 5)

#coordenadas x,y, largura e altura das as faces
for (x,y,w,h) in faces:
    #retangulo da face na imagem
    cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    #adicionando 2 regiões de interesse, uma em escala de cinza e uma de cores
    roi_gray = image_gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    #detecção dos olhos 
    eyes = eye_classifier.detectMultiScale(roi_gray)
    #coordenadas x,y, largura e altura dos olhos
    for (ex, ey, ew, eh) in eyes:
        #retangulo do olho na imagem (imagem/ponto de inicio/ponto final/cor/expressura do retangulo em pixel)
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,0,0),2)

cv.imshow('image',image)
cv.waitKey(0)
cv.destroyAllWindows()


