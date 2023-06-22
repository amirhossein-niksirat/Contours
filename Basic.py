import cv2
import numpy as np


class Image():
    def __init__(self, filename):
        self.filename = filename
        self.image = cv2.imread(self.filename)        

    @property
    def Height(self):
        return self.image.shape[0]

    @property
    def Width(self):
        return self.image.shape[1]

    @property
    def Mean(self):
        return np.mean(self.image)

    @property
    def Median(self):
        return np.median(self.image)

    @property
    def StdDev(self):
        return np.std(self.image)

    def imgOriginal(self):
        return cv2.imread(self.filename)

    def Negative(self):
        return 255 - self.image

    def Gray(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def vector(self, image=None):
        if image is None:
            image = self.image
        image = image.ravel()
        return image

    def BGR(self):
        return cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def YUV(self):
        img = cv2.imread(self.filename)
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    def HLS(self):
        img = cv2.imread(self.filename)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    def StdRange_In(self, image=None):
        if image is None:
            image = self.image
        Up = np.median(image) + np.std(image)
        Down = np.median(image) - np.std(image)
        inImg = image * ((image >= Down) & (image <= Up))
        return inImg

    def StdRange_Out(self, image=None):
        if image is None:
            image = self.image
        Up = np.median(image) + np.std(image)
        Down = np.median(image) - np.std(image)
        outImg = image * ((image <= Down) | (image >= Up))
        return outImg

    def Histogram(self, image=None):
        if image is None:
            image = self.image
        return np.int32(cv2.calcHist([image], [0], None, [256], [0, 256]))

    def Weight(self, image=None):
        if image is None:
            image = self.image
        weight = image / np.sum(image) 
        return weight       
         
    def WeightofHistogram(self, image=None):
        if image is None:
            image = self.image
        hist = np.int32(cv2.calcHist([image], [0], None, [256], [0, 256]))
        s = np.sum(hist)
        wHist = hist / s
        wHist = wHist[:, 0]
        w = image * wHist[image]
        return w

    def Show(self):
        cv2.imshow(self.filename, self.image)



###
img = Image("photo.jpg")

from Contour import ContourInfo

cnt = ContourInfo(img.Gray())
import cv2
cv2.imshow("LL", cnt.draw_contours())
