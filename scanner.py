import cv2
import numpy as np

modelo_atual = 'Nenhum'
status_barra = "AGUARDANDO"
max_porcentagem_falhas = 0
um_limiar_de_falhas = 40 # PORCENTAGEM DE FALHAS PARA IDENTIFICAR BARRA

video_path = 'static/video_esteira.mp4'
cap = cv2.VideoCapture(video_path)
pontos_roi = [(210, 20), (250, 20), (250, 100), (210, 100)]

def area_de_interesse(frame, pontos_roi):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [np.array(pontos_roi, np.int32)], (255, 255, 255))
    roi = extrair_roi_corretamente(frame, pontos_roi)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel_bgr = cv2.cvtColor(cv2.convertScaleAbs(sobel), cv2.COLOR_GRAY2BGR)
    return sobel_bgr

def extrair_roi_corretamente(frame, pontos_roi):
    pts = np.array(pontos_roi, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    roi_cortada = masked[y:y+h, x:x+w]
    return roi_cortada

def menu_lateral(frame, status, cor_status, porcentagem_falhas, resultado, cor_resultado, roi_falha=None, roi_falha_sobel=None):
    global modelo_atual
    altura, largura = frame.shape[:2]
    largura_menu = 300
    cor_fundo = (50, 50, 50)
    menu = np.full((altura, largura_menu, 3), cor_fundo, dtype=np.uint8)
    margem_superior = 30
    espaco_entre_textos = 30
    cv2.putText(menu, f"Status: {status}", (10, margem_superior), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_status, 1)
    cv2.putText(menu, f"Falhas: {porcentagem_falhas:.2f}%", (10, margem_superior + espaco_entre_textos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_status, 1)
    cv2.putText(menu, resultado, (10, margem_superior + 2*espaco_entre_textos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_resultado, 1)
    inicio_y = margem_superior + 3*espaco_entre_textos + 20
    espaco_entre_imagens = 10
    largura_imagem = (largura_menu - 3*espaco_entre_imagens) // 2
    altura_imagem = 100
    cv2.putText(menu, "Original", (10, inicio_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu, "Processada", (largura_imagem + 2*espaco_entre_imagens, inicio_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if roi_falha is not None:
        roi_resized = cv2.resize(roi_falha, (largura_imagem, altura_imagem))
        menu[inicio_y:(inicio_y + altura_imagem), espaco_entre_imagens:(espaco_entre_imagens + largura_imagem)] = roi_resized
    if roi_falha_sobel is not None:
        roi_sobel_resized = cv2.resize(roi_falha_sobel, (largura_imagem, altura_imagem))
        menu[inicio_y:(inicio_y + altura_imagem), 2*espaco_entre_imagens + largura_imagem:(2*espaco_entre_imagens + 2*largura_imagem)] = roi_sobel_resized
    frame_com_menu = np.hstack((frame, menu))
    return frame_com_menu

def process_frame(frame, pontos_roi):
    global status_barra, um_limiar_de_falhas, max_porcentagem_falhas, modelo_atual
    roi_falha = extrair_roi_corretamente(frame, pontos_roi)
    roi_falha_sobel = area_de_interesse(frame, pontos_roi)
    porcentagem_falhas_atual = np.mean(roi_falha_sobel) / 255 * 100
    status = "EM ANALISE" if porcentagem_falhas_atual < um_limiar_de_falhas else "AGUARDANDO..."

    if status == "EM ANALISE":
        max_porcentagem_falhas = max(max_porcentagem_falhas, porcentagem_falhas_atual)

    cor_status = (255, 255, 255)

    if max_porcentagem_falhas > 10:
        resultado = f"RECUSADO ({max_porcentagem_falhas:.2f}% de Falha)"
        cor_resultado = (0, 0, 255)
    else:
        resultado = f"APROVADO ({max_porcentagem_falhas:.2f}% de Falha)"
        cor_resultado = (0, 255, 0)

    pts = np.array(pontos_roi, dtype=np.int32).reshape((-1, 1, 2))
    x, y, w, h = cv2.boundingRect(pts)
    roi_resized = cv2.resize(roi_falha_sobel, (w, h))
    frame[y:y+h, x:x+w] = roi_resized
    frame_com_menu = menu_lateral(frame, status, cor_status, porcentagem_falhas_atual, resultado, cor_resultado, roi_falha, roi_falha_sobel)
    return frame_com_menu

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_processado = process_frame(frame, pontos_roi)
        cv2.imshow('Scanner Qualidade', frame_processado)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Sair
            break
        elif key == ord('1'):
            modelo_atual = 'Modelo 1'
        elif key == ord('2'):
            modelo_atual = 'Modelo 2'

cap.release()
cv2.destroyAllWindows()
