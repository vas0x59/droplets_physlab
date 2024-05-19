import cv2
import numpy as np
import pathlib
import pickle
import sys


path_dir = pathlib.Path(sys.argv[1])

l_kksize = 23
def float01_to_uint8(fl):
    return np.clip(fl*255, 0, 255).astype(np.uint8)
norm_f = None

def proc(img):
    global norm_f
    R_output = None
    debug = img.copy()
    # crop = img[track_point[1]:blink_point[1], track_point[0]:blink_point[0], :]
    cv2.imshow("img", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    if norm_f is None:
        norm_f = np.max(gray)
    gray/=norm_f
    cv2.imshow("gray", gray)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    lap = cv2.Laplacian(gray, ksize=l_kksize, ddepth=cv2.CV_32F)
    # lap = cv2.GaussianBlur(lap, (7,7), 0)
    lap = np.clip(lap/(2**(l_kksize*2 -2-2-2)), 0 , 200)
    # lap = cv2.morphologyEx(lap, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=l_kksize)/(2**(l_kksize*2 -1-0-2))
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=l_kksize)/(2**(l_kksize*2 -1-0-2))
    sobel_abs = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    # mask = (sobel_abs> 0.03).astype(np.uint8)*255
    mask = (lap > 0.005).astype(np.uint8)*255
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))

    cv2.imshow("sobel_abs", sobel_abs*10)
    cv2.imshow("mask", mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    # circles = []
    if hierarchy is not None:
        for i, (cnt, h) in enumerate(zip(contours, hierarchy[0])):
            a = cv2.contourArea(cnt)
            (x,y),radius = cv2.minEnclosingCircle(cnt)

            # print(a, np.pi*radius**2)
            if a > 200 and np.abs(a - np.pi*radius**2)/(np.pi*radius**2) < 0.1:   
                x,y,w,h = cv2.boundingRect(cnt) 
                # print(h)
                # h = cv2.convexHull(cnt)
                # cv2.drawContours(debug, [h], -1, (255,0,0), 2)
                # if cv2.contourArea(h) > a * 2:
                cnts.append(cnt)
                cv2.drawContours(debug, [cnt], -1, (0,i*255/len(contours),0), 2)
        if len(cnts) >=1:
            cnt = max(cnts, key=lambda c: cv2.contourArea(c))
            R = np.sqrt(cv2.contourArea(cnt)/np.pi)
            M = cv2.moments(cnt)
            # print(cnt.shape)
            x = (M["m10"] / (M["m00"]+1e-10))
            y = (M["m01"] / (M["m00"] + 1e-10))
            R_output = R
            # cv2.drawContours(debug, [], -1, (255,0,0), 5)
            cv2.circle(debug, (int(x), int(y)), int(R), (0, 0, 255), 5)

    # h_circles = cv2.HoughCircles(float01_to_uint8(gray),cv2.HOUGH_GRADIENT_ALT,1,5,
    #                 param1=50,param2=0.80,minRadius=20,maxRadius=500)
    # for hc in h_circles[0]:
    #     print(hc)
    #     cv2.circle(debug, (int(hc[0]), int(hc[1])), int(hc[2]), (255, 0, 0), 3)
    
    cv2.imshow("debug", debug)
    cv2.imshow("lap", lap*20)
    return R_output


print(path_dir)
images = [str(ip) for ip in path_dir.glob("*.jpg")]
images.sort(key=lambda ip: int(ip.split("_")[-1].split(".")[0]))

res = []

# norm_f = np.max(gray)

for p in  images:
    print(p)
    img = cv2.imread(str(p))
    img = cv2.resize(img, (0, 0),fx=0.5, fy=0.5)
    R_output = proc(img)
    if R_output is not None:
        res.append((p, R_output))
    cv2.waitKey(1)


print(res)
pickle.dump(res, open(f"{path_dir.name}_res.pickle", "wb"))
