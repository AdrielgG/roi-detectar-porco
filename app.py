import cv2
from bounding_box import bounding_box as bb
from datetime import datetime as dt
from farmertech.research.AI import CNN_Model
import numpy as np

cap = cv2.VideoCapture("Porco.mp4")

beberoudo = (10, 40, 240, 360)
#patio = (380,0,150,480)
cocho = (500, 200, 180, 250)

np.set_printoptions(suppress=True)

model1 = CNN_Model(keras_model='porcao.h5',roi=cocho)
model2 = CNN_Model(keras_model='porcao.h5',roi=beberoudo)
#model3 = CNN_Model(keras_model='porcao.h5',roi=patio)

def main():
   cont = 0
   while(True):
        cont+=1
        if not cont%4 == 0:
           continue
        ret, frame = cap.read()
        rcocho = model1.result(frame)
        rbeberoudo = model2.result(frame)
    #    rpatio = model3.result(frame)
        image = cv2.resize(frame,(int(720),int(480)))
        bb.add(image, cocho[0], cocho[1], cocho[0]+ cocho[2], cocho[1]+cocho[3], "COCHO STATUS: " + rcocho[0], "orange")
        bb.add(image, beberoudo[0], beberoudo[1], beberoudo[0]+ beberoudo[2], beberoudo[1]+beberoudo[3], "BEBEROUDO STATUS: " + rbeberoudo[0], "aqua")
    #    bb.add(image, patio[0], patio[1], patio[0]+ patio[2], patio[1]+patio[3], "PATIO STATUS: " + rpatio[0], "green")
        img = image
        #roi_cocho = result[1]
        #cv2.imshow("roi sensor",roi_cocho)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
