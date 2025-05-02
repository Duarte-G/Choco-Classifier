import cv2
import numpy as np

def extract_features(image):
    # 1) Cor: histograma HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    hist_color = np.concatenate([hist_h, hist_s, hist_v])
    hist_color = hist_color / np.sum(hist_color)

    # 2) Forma: contorno maior e medidas básicas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
    else:
        aspect_ratio = 0
        area = 0
        perimeter = 0

    shape_feats = np.array([aspect_ratio, area, perimeter])

    return np.concatenate([hist_color, shape_feats])