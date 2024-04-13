import cv2
import numpy as np
import time

video_path = 'static/video_esteira3.mp4'
cap = cv2.VideoCapture(video_path)
pontos_roi = [(660, 10), (350, 10), (350, 475), (660, 475)]
frames_processados = []
resultado = ''

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
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [np.array(pontos_roi, np.int32)], (255, 255, 255))
    gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel_bgr = cv2.cvtColor(cv2.convertScaleAbs(sobel), cv2.COLOR_GRAY2BGR)
    return sobel_bgr

def pontuacoes(processed_roi_contour):
    porcentagem_falhas_atual = np.mean(processed_roi_contour) / 255 * 100
    frames_processados.append(porcentagem_falhas_atual)
    return porcentagem_falhas_atual

def analise_de_frames(frames_processados_final):
    mini = min(frames_processados_final)
    maxi = max(frames_processados_final)
    avg = sum(frames_processados_final) / len(frames_processados_final)
    return f"{mini:.2f}%", f"{avg:.2f}%", f"{maxi:.2f}%"


def menu_lateral(frame, status, porcentagem_falhas_atual, resultado):
    cor_status=(255, 255, 255) if status=="OCIOSO" else (0, 255, 0)
    altura, largura = frame.shape[:2]
    largura_menu = 300
    cor_fundo = (50, 50, 50)
    menu = np.full((altura, largura_menu, 3), cor_fundo, dtype=np.uint8)
    margem_superior = 30
    espaco_entre_textos = 30
    cv2.putText(menu, f"Status: {status}", (10, margem_superior), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_status, 1)
    cv2.putText(menu, f"Falhas: {porcentagem_falhas_atual:.2f}%", (10, margem_superior + espaco_entre_textos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cor_resultado=(255, 255, 255) if status=="OCIOSO" else (0, 255, 0)
    cor_resultado=(255, 255, 255)
    cv2.putText(menu, str(resultado), (10, margem_superior + 2*espaco_entre_textos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_resultado, 1)
    inicio_y = margem_superior + 3*espaco_entre_textos + 20
    frame_com_menu = np.hstack((frame, menu))
    return frame_com_menu



while True:
    ret, frame = cap.read()
    time.sleep(0.05)
    if not ret:
        break
    roi, (rx, ry, rw, rh) = extrair_roi_corretamente(frame)  # Extract ROI and its position
    success, largest_rect = contornos(roi)  # Find the largest contour in the ROI
    if success:
        resultado = 'Processando...'
        status = "EM ANALISE"
        x, y, w, h = largest_rect
        roi_contour = roi[y:y+h, x:x+w]
        processed_roi_contour = process_frame(roi_contour)
        porcentagem_falhas_atual = pontuacoes(processed_roi_contour) # Coleta pontuação do frame
        roi[y:y+h, x:x+w] = processed_roi_contour
        frame[ry:ry+rh, rx:rx+rw] = roi
    elif len(frames_processados) > 0: # fechamento de frames analisados
        status = "FECHANDO..."
        resultado = analise_de_frames(frames_processados)
        frames_processados = []
    else:
        status = "OCIOSO"
    frame_com_menu = menu_lateral(frame, status, porcentagem_falhas_atual, resultado)
    cv2.imshow('Frame', frame_com_menu)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
