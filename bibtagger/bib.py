import cv2
import numpy as np

from . import bibtagger as bt
from . import find_bibs as bf


class Bib(object):
    def __init__(self, image, bodybox, outdir):
        self.image = image
        self.bodybox = bodybox
        self.outdir = outdir
        self.corners = bf.find_bib(self.body_image(), outdir)
        x, y, w, h = cv2.boundingRect(self.corners)
        self.bib_found = (x != 0 and y != 0 and w != 1 and h != 1)
        self.number = None

    def has_bib_number(self):
        return self.number != None and self.number != ''

    def body_image(self):
        return bt.getSubImage(self.image, self.bodybox)

    def body_image_with_bib(self):
        img = np.copy(self.body_image())
        cv2.drawContours(img, [self.corners], -1, (0, 0, 255), 2)
        return img

    def corners_relative_to_main_image(self):
        x_delta = self.bodybox[0]
        y_delta = self.bodybox[1]
        return np.array([[(pt[0][0] + x_delta, pt[0][1] + y_delta)] for pt in self.corners])

    def smallest_subimage_containing_bib(self):
        if not self.bib_found:
            return self.body_image()

        return bt.getSubImage(self.body_image(), cv2.boundingRect(self.corners))
