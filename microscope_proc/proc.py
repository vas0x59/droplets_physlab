from list_experiments import list_experiments


import cv2
import numpy as np
import glob
import math

import matplotlib.pyplot as plt
path00 = "/Users/vasily/Documents/droplets/untitled folder"

for path in glob.glob(path00 + "/*"):
    photos_exps =  list_experiments(path)

    DEBUG = True

    droplets_all_exp = []

    for n, ps in photos_exps:
        droplets_local = []
        for i, p in enumerate(ps):
            droplets = []
        # i, p = next(enumerate(ps))
            bgr = cv2.imread(p)
            debug = bgr.copy()
            g_kksize = 13
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
            blur = cv2.GaussianBlur(gray, (g_kksize,g_kksize), 0)
            if DEBUG:
                cv2.imshow("bgr", bgr)
            l_kksize= 7
            # lap = cv2.Laplacian(blur, ksize=l_kksize, ddepth=cv2.CV_32F)
            # lap = np.clip(lap/(2**(l_kksize*2 -2-2-2)), 0 , 1)

            dx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=l_kksize)/(2**(l_kksize*2 -1-0-2))
            dy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=l_kksize)/(2**(l_kksize*2 -1-0-2))
            sobel_abs = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
            edge = sobel_abs
            mask = (edge > 0.01).astype(np.uint8)
            
            # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))

            if DEBUG:
                cv2.imshow("edge", (edge) *20)
                # cv2.imshow("sobel_abs", (sobel_abs) *20)
                
                # debug[mask==1, 0] = 255

                cv2.imshow("mask", mask*255)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #          

            for i, cnt in enumerate(contours):
                hr  = hierarchy[0, i]
                x,y,w,h = cv2.boundingRect(cnt)
                pnt = np.array([x+w/2, y+h/2])
                if hr[3] == -1 and w > 2 and h > 2 and np.linalg.norm(pnt - np.array([*gray.shape[::-1]])/2)< gray.shape[1]*0.5:
                    crop_edge = edge[int(y+h*0.1):int(math.ceil(y+h*0.9)), int(x+w*0.1):int(math.ceil(x+w*0.9))]
                    crop_gray = gray[int(y+h*0.1):int(math.ceil(y+h*0.9)), int(x+w*0.1):int(math.ceil(x+w*0.9))]
                    

                    qe90 = np.quantile(crop_edge, 0.9)
                    # qe50 = np.quantile(crop_edge, 0.5)
                    # qe10 = np.quantile(crop_edge, 0.1)

                    # qg90 = np.quantile(crop_gray, 0.9)
                    qg50 = np.quantile(crop_gray, 0.5)
                    qg10 = np.quantile(crop_gray, 0.1)
                    # q10  = np.quantile(crop, 0.1)
                    if qe90 > 0.008 and qg50 < 0.15 and  abs(w-h) < min(w, h)*0.7 and qg10 < 0.1:
                        cv2.rectangle(debug,(x,y),(x+w,y+h),(0,255,0),2)
                        # cv2.putText(debug, f"{qg10:.2f} {qg50:.2f} {qg90:.2f} {qe10:.2f} {qe50:.2f} {qe90:.2f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
                        # area = cv2.contourArea(cnt)
                        R = np.sqrt(w*h)
                        pnt = (x+w/2, y+h/2)
                        droplets.append((p, R))

            

            print(hierarchy)
            # cv2.drawContours(debug, contours, -1, (0,0,255), -1)
            if DEBUG:
                cv2.imshow("debug", debug)
                # cv2.imshow("ln lap", np.log(lap))
                k = cv2.waitKey(1)
                if ord("q") == k:
                    exit(0)
            droplets_local.extend(droplets)
        
        plt.figure()
        plt.hist(np.array([d[1] for d in droplets_local]), bins=np.linspace(0, 90, 30))
        plt.savefig(path + "/" + n + "_plt.png")
        droplets_all_exp.extend(droplets_local)
    Rs = np.array([d[1] for d in droplets_all_exp])
    np.save(path+ "/Rs.npy", Rs)
    plt.figure()
    plt.hist(Rs, bins=np.linspace(0, 50, 25))
    plt.savefig(path + "/plt.png")