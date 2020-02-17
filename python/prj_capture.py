import cv2
import sys
import os
import time
import numpy as np

def store_imgs(path, imgs):
    os.makedirs(path, exist_ok=True)

    for i in enumerate(imgs):
        cv2.imwrite(path + "/" + str(i[0]) + ".png", i[1])

def load_imgs(path):
    imgs = []
    files = []
    for e in os.scandir(path):
        if e.is_file() == False:
            continue
        files.append(e.path)
    files = sorted(files)

    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        imgs.append(img)

    return imgs

def project_pattern(win_name, pattern, period):
    cv2.imshow(win_name, pattern)
    cv2.waitKey(period)

def make_capture(cam):
   cv2.waitKey(1000)
   ret, frame = cam.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   fgray = np.asarray(gray, dtype="float32")

   return fgray

def make_captures(win_name, cam, patterns, period):
    captures = []
    for p in patterns:
        project_pattern(win_name, p, period)
        cap_img = make_capture(cam)
        cap_img = make_capture(cam)
        captures.append(cap_img)

    return captures

def create_cam():
    return cv2.VideoCapture(0)

def create_window(name):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN)

    return name

patterns_path = sys.argv[1]
projections_path = sys.argv[2]
period = int(sys.argv[3])

cam = create_cam()
win = create_window('projector')

time.sleep(3)
patterns = load_imgs(patterns_path)
projections = make_captures(win, cam, patterns, period)
store_imgs(projections_path, projections)

cv2.destroyAllWindows()
