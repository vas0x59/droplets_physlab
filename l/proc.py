import cv2
import numpy as np
import pickle 

import glob
from datetime import datetime
# path = "/Users/vas ily/Downloads/Telegram Desktop/VID_20240410_204750.mp4"


import sys
path = sys.argv[1]
display = True

images = [ip for ip in glob.glob(path + "/*")]
images.sort(key=lambda ip: int(ip.split("_")[-1].split(".")[0]))
print(images)
def read_img(num):
    global images
    return cv2.resize(cv2.imread(images[num]), (0, 0), fx=0.5, fy=0.5)


def _nt(p: np.ndarray) -> tuple:
    return (int(p[0]), int(p[1]))
# cap = cv2.VideoCapture(path)

track_point = np.zeros(2,dtype=int)
blink_point = np.ones(2,dtype=int)*3
init_point = np.zeros(2, dtype=int)
current_pnt = np.zeros(2, dtype=int)
init_mode = False
play = False

pid = False

def callback(event, x, y, flags, params):
    global blink_point, track_point, init_mode, play
    # print(pnts, selected_i, event)
    if not play:
        if not init_mode:
            if pid:
                if event == cv2.EVENT_RBUTTONDOWN:
                    blink_point[0] = x
                    blink_point[1] = y
            if not pid:
                if event == cv2.EVENT_RBUTTONDOWN:
                    track_point[0] = x
                    track_point[1] = y
        else:
            if event == cv2.EVENT_RBUTTONDOWN:
                init_point[0] = x
                init_point[1] = y


def proc(img):
    global current_pnt
    crop = img[track_point[1]:blink_point[1], track_point[0]:blink_point[0], :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    # # blur = cv2.GaussianBlur(gray, (g_kksize,g_kksize), 0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=l_kksize)/(2**(l_kksize*2 -1-0-2))
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=l_kksize)/(2**(l_kksize*2 -1-0-2))
    sobel_abs = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    maskb = sobel_abs > 0.01

    ys, xs = np.where(maskb)

    pnts = np.zeros((len(ys), 2))
    pnts[:, 0] = xs
    pnts[:, 1] = ys
    pnts  = pnts
    cur_pnts = pnts[np.linalg.norm(pnts - current_pnt, axis=1) <  50, :]
    
    

    
    update_pnt = np.mean(cur_pnts, axis=0)
    print(update_pnt)

    for p in cur_pnts:
        cv2.circle(crop, _nt(p), 1, (0, 0, 255), -1)

    ab = None
    
    if update_pnt is not None and len(update_pnt) > 0 and not np.isnan(update_pnt[0]):
        # cv2.circle(img, _nt(update_pnt + track_point), 10, (0, 0, 255), 4)
        if play:
            current_pnt = update_pnt.copy()
        # q = 0.08
        # x1 = np.quantile(cur_pnts[:, 0], q).astype(int)
        # x2 = np.quantile(cur_pnts[:, 0], 1-q).astype(int)
        # y1 = np.quantile(cur_pnts[:, 1], q).astype(int)
        # y2 = np.quantile(cur_pnts[:, 1], 1-q).astype(int)
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) 
        x = cur_pnts[:, 0]
        y = cur_pnts[:, 1]
        A = np.vstack([x**2, x * y, y**2, x, y]).T
        b = np.ones((len(x), 1))
        x = (np.linalg.pinv(A) @ b ).flatten()

        A, B, C, D, E = -x
        F = 1
        a = -np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)*((A+C) + np.sqrt((A-C)**2 +B**2)) )/(B**2 - 4*A*C)
        b = -np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)*((A+C) - np.sqrt((A-C)**2 +B**2)) )/(B**2 - 4*A*C)
        ab = (a, b)
        # if play:
        # ab_s.append((a,b))
        # print(a*b*min(a,b) *np.pi *4/3)
        # for p in fit:
            # cv2.circle(crop, _nt(p), 1, (255, 0, 0), -1)





    cv2.circle(img, _nt(current_pnt + track_point), 10, (255, 0, 0), 4)
    if display:
        cv2.imshow("cimg", img)
        cv2.imshow("crop", crop)
        cv2.imshow("sobel_abs", maskb*sobel_abs*20)
    return ab


# ret, frame = cap.read()
cv2.namedWindow("cimg")
# cv2.waitKey(1)
cv2.setMouseCallback("cimg", callback)

i = 0
# g_kksize = 11
l_kksize = 11



img0 = read_img(0)

# N = 600





while True:
    
    img = img0.copy()
    # cv2.imshow()
    # proc(img)

    cv2.circle(img, _nt(track_point), 10, (255, 0, 255), 8)
    cv2.circle(img, _nt(blink_point), 10, (255, 255, 0), 8)
    cv2.circle(img, _nt(init_point), 10, (0, 255, 0), 4)
    cv2.imshow("cimg", img)
    # if not ret:
        # break
    k = cv2.waitKey(10)
    if k == ord("q"):
        break
    if k == ord("p"):
        play = True
        break
    if k == ord("i"):
        init_mode = not init_mode
    if k == ord("t"):
        pid = not pid
    if k == ord("s"):
        pickle.dump((track_point, blink_point), open("points.pickle", "wb"))
        pickle.dump(init_point, open("init_point.pickle", "wb"))
    if k == ord("l"):
        (track_point, blink_point) = pickle.load(open("points.pickle", "rb"))
        track_point = track_point.astype(int)
        blink_point = blink_point.astype(int)
        init_point = pickle.load(open("init_point.pickle", "rb"))
        current_pnt = init_point.copy() - track_point
ab_s = []

for i in range(len(images)):
    img = read_img(i)
    ab = proc(img)
    
    if cv2.waitKey(1) == ord("z"):
        display =  not display
    if ab is not None:
        ab_s.append([datetime.strptime(images[i].split("_")[-1].split(".")[0], "%H%M%S").timestamp(), *ab])

# Ts = np.array([ for i in ab_s])
# As = 

pickle.dump(ab_s, open("ab_s.pickle", "wb"))