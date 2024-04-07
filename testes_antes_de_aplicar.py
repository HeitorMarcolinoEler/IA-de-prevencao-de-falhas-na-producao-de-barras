import cv2
import numpy as np
import time

video_path = 'static/video_esteira3.mp4'
cap = cv2.VideoCapture(video_path)
pontos_roi = [(660, 10), (350, 10), (350, 475), (660, 475)]

def extrair_roi_corretamente(frame):
    pts = np.array(pontos_roi, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    roi_cortada = masked[y:y+h, x:x+w]
    return roi_cortada, (x, y, w, h)  # Return both the ROI and its bounding box

def contornos(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 12000
    largest_rect = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            x, y, w, h = cv2.boundingRect(contour)
            largest_rect = (x, y, w, h)
    if largest_rect is not None:
        return True, largest_rect
    else:
        return False, None

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
    return frame

def area_de_interesse(frame):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [np.array(pontos_roi, np.int32)], (255, 255, 255))
    roi = extrair_roi_corretamente(frame)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel_bgr = cv2.cvtColor(cv2.convertScaleAbs(sobel), cv2.COLOR_GRAY2BGR)
    return sobel_bgr

while True:
    ret, frame = cap.read()
    time.sleep(0.05)
    if not ret:
        break
    roi, (rx, ry, rw, rh) = extrair_roi_corretamente(frame)  # Extract ROI and its position
    success, largest_rect = contornos(roi)  # Find the largest contour in the ROI
    if success:
        x, y, w, h = largest_rect
        roi_contour = roi[y:y+h, x:x+w]
        # roi_contour = area_de_interesse(roi_contour)
        processed_roi_contour = process_frame(roi_contour)
        roi[y:y+h, x:x+w] = processed_roi_contour
        frame[ry:ry+rh, rx:rx+rw] = roi
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
