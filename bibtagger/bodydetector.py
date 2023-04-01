import numpy as np
import cv2
import math
import sys
import os

from imutils.object_detection import non_max_suppression

# Original code from : https://gitlab.com/wavexx/facedetect
# facedetect: a simple face detector for batch processing
# Copyright(c) 2013-2017 by wave++ "Yuri D'Elia" <wavexx@thregr.org>
# Distributed under GPLv2+ (see COPYING) WITHOUT ANY WARRANTY.

# Profiles
DATA_DIR = './'
CASCADES = {}

PROFILES = {
    'HAAR_FRONTALFACE_ALT2': 'haarcascades/haarcascade_frontalface_alt2.xml',
}


# Face normalization
NORM_SIZE = 100
NORM_MARGIN = 10

# Support functions
def error(msg):
    sys.stderr.write("{}: error: {}\n".format(os.path.basename(sys.argv[0]), msg))


def fatal(msg):
    error(msg)
    sys.exit(1)


def load_cascades(data_dir):
    for k, v in PROFILES.items():
        v = os.path.join(data_dir, v)
        try:
            if not os.path.exists(v):
                raise cv2.error('no such file')
            CASCADES[k] = cv2.CascadeClassifier(v)
        except cv2.error:
            fatal("cannot load {} from {}".format(k, v))


def crop_rect(im, rect, shave=0):
    return im[rect[1]+shave:rect[1]+rect[3]-shave,
              rect[0]+shave:rect[0]+rect[2]-shave]


def shave_margin(im, margin):
    return im[margin:-margin, margin:-margin]


def norm_rect(im, rect, equalize=True, same_aspect=False):
    roi = crop_rect(im, rect)
    if equalize:
        roi = cv2.equalizeHist(roi)
    side = NORM_SIZE + NORM_MARGIN
    if same_aspect:
        scale = side / max(rect[2], rect[3])
        dsize = (int(rect[2] * scale), int(rect[3] * scale))
    else:
        dsize = (side, side)
    roi = cv2.resize(roi, dsize, interpolation=cv2.INTER_CUBIC)
    return shave_margin(roi, NORM_MARGIN)


def rank(im, rects):
    scores = []
    best = None

    for i in range(len(rects)):
        rect = rects[i]
        roi_n = norm_rect(im, rect, equalize=False, same_aspect=True)
        roi_l = cv2.Laplacian(roi_n, cv2.CV_8U)
        e = np.sum(roi_l) / (roi_n.shape[0] * roi_n.shape[1])

        dx = im.shape[1] / 2 - rect[0] + rect[2] / 2
        dy = im.shape[0] / 2 - rect[1] + rect[3] / 2
        d = math.sqrt(dx ** 2 + dy ** 2) / (max(im.shape) / 2)

        s = (rect[2] + rect[3]) / 2
        scores.append({'s': s, 'e': e, 'd': d})

    sMax = max([x['s'] for x in scores])
    eMax = max([x['e'] for x in scores])

    for i in range(len(scores)):
        s = scores[i]
        sN = s['sN'] = s['s'] / sMax
        eN = s['eN'] = s['e'] / eMax
        f = s['f'] = eN * 0.7 + (1 - s['d']) * 0.1 + sN * 0.2

    ranks = range(len(scores))
    ranks = sorted(ranks, reverse=True, key=lambda x: scores[x]['f'])
    for i in range(len(scores)):
        scores[ranks[i]]['RANK'] = i

    return scores, ranks[0]


def mssim_norm(X, Y, K1=0.01, K2=0.03, win_size=11, sigma=1.5):
    C1 = K1 ** 2
    C2 = K2 ** 2
    cov_norm = win_size ** 2

    ux = cv2.GaussianBlur(X, (win_size, win_size), sigma)
    uy = cv2.GaussianBlur(Y, (win_size, win_size), sigma)
    uxx = cv2.GaussianBlur(X * X, (win_size, win_size), sigma)
    uyy = cv2.GaussianBlur(Y * Y, (win_size, win_size), sigma)
    uxy = cv2.GaussianBlur(X * Y, (win_size, win_size), sigma)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    A1 = 2 * ux * uy + C1
    A2 = 2 * vxy + C2
    B1 = ux ** 2 + uy ** 2 + C1
    B2 = vx + vy + C2
    D = B1 * B2
    S = (A1 * A2) / D

    return np.mean(shave_margin(S, (win_size - 1) // 2))


def face_detect(im, biggest=False):
    side = math.sqrt(im.size)
    minlen = int(side / 20)
    maxlen = int(side / 2)
    flags = cv2.CASCADE_DO_CANNY_PRUNING

    # optimize queries when possible
    if biggest:
        flags |= cv2.CASCADE_FIND_BIGGEST_OBJECT

    # frontal faces
    cc = CASCADES['HAAR_FRONTALFACE_ALT2']
    features = cc.detectMultiScale(im, 1.1, 4, flags, (minlen, minlen), (maxlen, maxlen))
    return features


def face_detect_file(path, biggest=False):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        fatal("cannot load input image {}".format(path))
    im = cv2.equalizeHist(im)
    features = face_detect(im, biggest)
    return im, features


def pairwise_similarity(im, features, template, **mssim_args):
    template = np.float32(template) / 255
    for rect in features:
        roi = norm_rect(im, rect)
        roi = np.float32(roi) / 255
        yield mssim_norm(roi, template, **mssim_args)

def getbodyboxes(image):
    #in: numpy image
    #out: list [(x,y,width,height)]

    faces = findfaces(image)

    bodyrectangles = findbodies(image,faces)

    return bodyrectangles


def findfaces(image):
    load_cascades("./bibtagger")
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = cv2.equalizeHist(im)
    features = face_detect(im)
    return features

def scale_rect(rect, scale):
    return [int(value*scale) for value in rect]

def findbodies(image, faces):

    bodies = np.zeros_like(faces)
    bodiesindex = 0

    #for each face, draw a body
    for (x, y, facewidth, faceheight) in faces:
        #3*faceheight, 7/3 * facewidth, .5*faceheight below the face.
        bodyheight = 4 * faceheight
        bodywidth = 7/3 * facewidth
        y_body = y + faceheight + .5 * faceheight
        x_body = x + .5 * facewidth - .5 * bodywidth

        bodies[bodiesindex] = (x_body,y_body, bodywidth, bodyheight)
        bodiesindex = bodiesindex + 1
        bodies = bodies.clip(min=0)
        #cv2.rectangle(image, (x_body, y_body), (x_body+bodywidth, y_body+bodyheight), (0, 255, 0), 2)

    return bodies

