from list_experiments import list_experiments


import cv2
import numpy as np
import glob

import matplotlib.pyplot as plt
path00 = "/Users/vasily/Downloads/Telegram Desktop/68 2"

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
            g_kksize = 11
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)/255.
            blur = cv2.GaussianBlur(gray, (g_kksize,g_kksize), 0)
            if DEBUG:
                cv2.imshow("bgr", bgr)
            l_kksize= 11
            lap = cv2.Laplacian(blur, ksize=l_kksize, ddepth=cv2.CV_64F)
            lap = np.clip(lap/(2**(l_kksize*2 -2-2-2)), 0 , 1)
            
            mask = (lap > 0.005).astype(np.uint8)
            
            # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))

            if DEBUG:
                cv2.imshow("lap", (lap*mask) *20)
                
                # debug[mask==1, 0] = 255

                cv2.imshow("mask", mask*255)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #          

            for i, cnt in enumerate(contours):
                h  = hierarchy[0, i]
                # if h[1] == -1:
                x,y,w,h = cv2.boundingRect(cnt)
                crop_lap = lap[y:y+h, x:x+w]
                crop_gray = gray[y:y+h, x:x+w]
                ql90 = np.quantile(crop_lap, 0.9)
                qg10 = np.quantile(crop_gray, 0.1)
                # q10  = np.quantile(crop, 0.1)
                if ql90 > 0.008 and  abs(w-h) < min(w, h)*0.7 and qg10 < 0.2:
                    cv2.rectangle(debug,(x,y),(x+w,y+h),(0,255,0),2)

                    # area = cv2.contourArea(cnt)
                    R = np.sqrt(w*h)
                    pnt = (x+w/2, y+h/2)
                    droplets.append((p, R))

            

            print(hierarchy)
            # cv2.drawContours(debug, contours, -1, (0,0,255), -1)
            if DEBUG:
                cv2.imshow("debug", debug)
                # cv2.imshow("ln lap", np.log(lap))
                k = cv2.waitKey(0)
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