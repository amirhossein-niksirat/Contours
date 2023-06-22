import cv2
import numpy as np


class ContourInfo:
    def __init__(self, image, contours=None):
        self.image = image
        if contours is None:
            ret, thresh = cv2.threshold(image, np.mean(image), 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours

    def Contours(self, image=None, threshold=None):
        if image is None:
            image = self.image
        if threshold is None:
            threshold = np.mean(image)
        ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        i, cnt, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return np.array(cnt)

    def draw_contours(self, image=None, contours=None):
        if image is None:
            image = self.image
        if contours is None:
            contours = self.contours
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image, contours, -1, (100, 255, 0), 1)
        return image

    # Moments
    def moments(self, contours=None):
        if contours is None:
            contours = self.contours
        return cv2.moments(contours)

    # Center of Contours
    def center(self, contours=None):
        if contours is None:
            contours = self.contours
        M = cv2.moments(contours)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        contourCenter = (int(cx), int(cy))
        return contourCenter

    # Area of Contours
    def area(self, contours=None):
        if contours is None:
            contours = self.contours
        return cv2.contourArea(contours)

    # Perimeter of Contours
    def perimeter(self, contours=None):
        if contours is None:
            contours = self.contours
        return cv2.arcLength(contours, True)

    # Contour Approximation
    def approximation(self, contours=None):
        if contours is None:
            contours = self.contours
        epsilon = 0.1 * cv2.arcLength(contours, True)
        approx = cv2.approxPolyDP(contours, epsilon, True)
        return epsilon, approx

    # Convex Hull
    # hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
    def hull(self, contours=None):
        if contours is None:
            contours = self.contours
        return cv2.convexHull(contours)

    # Checking Convexity
    def isConvex(self, contours=None):
        if contours is None:
            contours = self.contours
        return cv2.isContourConvex(contours)

    # Bounding Rectangle
    # Straight Bounding Rectangle
    def bounding_rect(self, contours=None):
        if contours is None:
            contours = self.contours
        x, y, w, h = cv2.boundingRect(contours)
        return int(x), int(y), int(w), int(h)

    def draw_bounding_rect(self, image=None, contours=None):
        if image is None:
            image = self.image
        if contours is None:
            contours = self.contours
        x, y, w, h = cv2.boundingRect(contours)
        return cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Rotated Rectangle
    def bounding_rect_rotated(self, contours=None):
        if contours is None:
            contours = self.contours
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

    def draw_bounding_rect_rotated(self, image=None, contours=None):
        if image is None:
            image = self.image
        if contours is None:
            contours = self.contours
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
        return img

    # Minimum Enclosing Circle
    def bounding_circle(self, contours=None):
        print(len(contours))
        if contours is None:
            contours = self.contours
        (x, y), radius = cv2.minEnclosingCircle(contours)
        return x, y, radius

    def draw_bounding_circle(self, image=None, contours=None):
        if image is None:
            image = self.image
        if contours is None:
            contours = self.contours
        (x, y), radius = cv2.minEnclosingCircle(contours)
        cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 1)
        return image

    # Fitting an Ellipse
    def bounding_ellipse(self, contours=None):
        if contours is None:
            contours = self.contours
        return cv2.fitEllipse(contours)

    def draw_bounding_ellipse(self, image=None, contours=None):
        if image is None:
            image = self.image
        if contours is None:
            contours = self.contours
        ellipse = cv2.fitEllipse(contours)
        cv2.ellipse(image, ellipse, (0, 0, 255), 1)
        return image

    # Fitting a Line
    def fitLine(self, contours=None):
        if contours is None:
            contours = self.contours
        rows, cols = contours.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(contours, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        return rows, cols, lefty, righty

    # Aspect Ratio
    def aspectRatio(self, contours=None):
        if contours is None:
            contours = self.contours
        x, y, w, h = cv2.boundingRect(contours)
        return float(w) / h

    # Extent
    def extent(self, contours=None):
        if contours is None:
            contours = self.contours
        area = cv2.contourArea(contours)
        x, y, w, h = cv2.boundingRect(contours)
        rect_area = w * h
        return float(area) / rect_area

    # Solidity
    def solidity(self, contours=None):
        if contours is None:
            contours = self.contours
        area = cv2.contourArea(contours)
        h = cv2.convexHull(contours)
        hull_area = cv2.contourArea(h)
        return float(area) / hull_area

    # Equivalent Diameter
    def equivalent_Diameter(self, contours=None):
        if contours is None:
            contours = self.contours
        area = cv2.contourArea(contours)
        return np.sqrt(4 * area / np.pi)

    # Orientation
    def orientation(self, contours=None):
        if contours is None:
            contours = self.contours
        (x, y), (MA, ma), angle = cv2.fitEllipse(contours)
        return (x, y), (MA, ma), angle;

    # Mask and Pixel Points
    def mask(self, image=None, contours=None):
        if image is None:
            image = self.image
        if contours is None:
            contours = self.contours
        m = np.zeros(image.shape, np.uint8)
        cv2.drawContours(m, [contours], 0, 255, -1)
        return m

    # Maximum Value, Minimum Value and their locations
    def minMaxLoc(self, image=None):
        if image is None:
            image = self.image
        _mask = np.zeros(image.shape, np.uint8)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image, mask=_mask)
        return min_val, max_val, min_loc, max_loc

    # Mean Color or Mean Intensity
    def mean(self, image=None):
        if image is None:
            image = self.image
        _mask = np.zeros(image.shape, np.uint8)
        return cv2.mean(image, mask=_mask)

    # Extreme Points
    def extremePoints(self, contours=None):
        if contours is None:
            contours = self.contours
        leftmost = tuple(contours[contours[:, :, 0].argmin()][0])
        rightmost = tuple(contours[contours[:, :, 0].argmax()][0])
        topmost = tuple(contours[contours[:, :, 1].argmin()][0])
        bottommost = tuple(contours[contours[:, :, 1].argmax()][0])
        return leftmost, rightmost, topmost, bottommost

    # Convexity Defects
    def convexDefect(self, contours=None):
        if contours is None:
            contours = self.contours
        h = cv2.convexHull(contours)  # , returnPoints=False)
        return cv2.convexityDefects(contours, h)

    # Point Polygon Test
    def distanc(self, x, y, contours=None):
        if contours is None:
            contours = self.contours
        return cv2.pointPolygonTest(contours, (x, y), False)
        # if True Returns Distance, If False returns in or out contour +1,0,-1

    def contour_kmean(self, size, contours=None):
        if contours is None:
            contours = self.contours
        filtered_contour = []
        for i in range(len(contours)):
            if float(cv2.contourArea(contours[i]) > float(size)):
                filtered_contour.append(contours[i])
        return filtered_contour




