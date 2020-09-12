import argparse
import logging
import time
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
from scipy.spatial import distance
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
fps_time = 0

args= {'resize':'432x368', 'model':'mobilenet_thin','camera':0,'resize_out_ratio':4.0,'vid':"output.avi"}
w, h = model_wh(args['resize'])

if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args['model']), target_size=(w, h))
else:
    e = TfPoseEstimator(get_graph_path(args['model']), target_size=(432, 368))

cam = cv2.VideoCapture(args['camera'])
ret_val, image = cam.read()
cam_store = cv2.VideoCapture(args['vid'])
ret_val_store, image_store = cam_store.read()

while True:
    dist = True 
    
    ret_val_store, image_store = cam_store.read()
    print("DIST--CHANGED--::::")
    while dist:
        ret_val, image = cam.read()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args['resize_out_ratio'])
        #print("ZEROS::",image.shape)

        image, blackImage = TfPoseEstimator.draw_humans(image, humans, imgcopy=True,black_image = True)
        # import matplotlib.pyplot as plt
        # plt.imsave('test2.jpg', image, cmap='Greys')
        
        final = cv2.bitwise_or(image, image_store)
        cv2.imshow('tf-pose-estimation result', final)
        #print(np.nonzero(image_store).shape)
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        #cv2.imshow('tf-pose-estimation result', final)
        Aflat = blackImage.flatten(order='C')
        Bflat = image_store.flatten(order='C')

        dist = distance.cosine(Aflat, Bflat)
        print("DIST::",dist)
        if dist > 0.80:
            dist = False
        
        fps_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    if cv2.waitKey(25) & 0xFF == ord('q'): 
            break

cv2.destroyAllWindows()