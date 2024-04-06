import cv2
import numpy as np
import time


video_path = 'static/video_esteira3.mp4'
cap = cv2.VideoCapture(video_path)
pontos_roi = [(660, 10), (350, 10), (350, 475), (660, 475)]

def extrair_roi_corretamente(frame):
    # interessante recortar pois em alguns pontos da filmagem pode ter pouca ou muita iluminação
    # assim influenciando no resultado
    pts = np.array(pontos_roi, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    roi_cortada = masked[y:y+h, x:x+w]
    return roi_cortada

def contornos(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 12000  # Tamanho maximo da area
    largest_rect = None  # Armazena a maior area do momento
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            x, y, w, h = cv2.boundingRect(contour)
            largest_rect = (x, y, w, h)
    if largest_rect is not None: # Se uma area maior for encontrada, atualiza
        x, y, w, h = largest_rect
        return True, frame[y:y+h, x:x+w] # Retorna frame cortado na area identificada
    else:
        return False, frame # Retorna frame Default (nao foi encontrado contorno)
    
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
    return frame


while True:
    ret, frame = cap.read()
    time.sleep(0.05)
    if not ret:
        break
    frame = extrair_roi_corretamente(frame) # Corta area de interesse
    success, frame = contornos(frame)
    if success:
        frame = process_frame(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
