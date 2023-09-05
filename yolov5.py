import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os


IMAGES_PATH = os.path.join('data', 'images') #/data/images
labels = ['happy', 'angry']
number_imgs = 5

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/last.pt', force_reload=True)

img = os.path.join('data', 'images', 'happy.3a61fb6d-4bd3-11ee-9ea7-50ebf64365fd.jpg')
results = model(img)
results.print()

plt.imshow(np.squeeze(results.render()))
plt.show()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


