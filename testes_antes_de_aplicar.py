import cv2
import numpy as np
import time

video_path = 'static/video_esteira2_editado.mp4'
# video_path = 'static/video_esteira2.mp4'
# video_path = 'static/video_esteira3.mp4'
cap = cv2.VideoCapture(video_path)
pontos_roi = [(660, 10), (350, 10), (350, 475), (660, 475)]
threshold_media_resultado = 11
maior_area = 150000 # area maxima de contorno dinamico
area_limite = 80000 # area minima de contorno dinamico
frames_processados = []
resultados = []

def extrair_roi_corretamente(frame):
    pts = np.array(pontos_roi, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    roi_cortada = masked[y:y+h, x:x+w]
    return roi_cortada, (x, y, w, h)

def contornos(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_rect = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area_limite < area < maior_area:
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

def find_high_deviant_sequences(data, num_std_dev=1.2):
    # funcao para identificar sequencias de porcentagem mais altas que a media
    # identifica pontos de anormalidade durante o scanner
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    upper_bound = mean + num_std_dev * std_dev
    high_deviant_sequences = []
    current_sequence = []
    for value in data:
        if value > upper_bound:
            current_sequence.append(value)
        else:
            if current_sequence:  # If current sequence ends, add it to the list
                high_deviant_sequences.append(current_sequence)
                current_sequence = []
    if current_sequence:
        high_deviant_sequences.append(current_sequence)
    return high_deviant_sequences

def find_subsequence_indexes(lista_geral, lista_anormalidade): # encontra sequencia com porcentagem anormal dentro da analise geral
    for i in range(len(lista_geral) - len(lista_anormalidade) + 1):
        if lista_geral[i:i+len(lista_anormalidade)] == lista_anormalidade:
            return i, i+len(lista_anormalidade)-1
    return -1, -1

def analise_de_frames(frames_processados_final):
    lista_pontuacao_anormal = find_high_deviant_sequences(frames_processados_final, num_std_dev=1.2)
    sequencia_anormal = 'OK'
    for pontuacao_anormal in lista_pontuacao_anormal:
        if pontuacao_anormal == frames_processados_final[:len(pontuacao_anormal)]:
            frames_processados_final = frames_processados_final[len(pontuacao_anormal):] # remove falso positivo começo
        elif pontuacao_anormal == frames_processados_final[-len(pontuacao_anormal):]:
            frames_processados_final = frames_processados_final[:len(frames_processados_final)-len(pontuacao_anormal)] # remove falso positivo final
        if len(pontuacao_anormal) > 2: # pega sequencias de anormalidade
            sequencia_anormal = find_subsequence_indexes(frames_processados_final, pontuacao_anormal)
    mini = min(frames_processados_final)
    maxi = max(frames_processados_final)
    avg = sum(frames_processados_final) / len(frames_processados_final)
    return f"{mini:.2f}%", f"{avg:.2f}%", f"{maxi:.2f}%", len(frames_processados_final), sequencia_anormal

def escolhe_cor_resultado(resultado):
    if resultado[4] != 'OK' and float(resultado[1][:-1])>threshold_media_resultado: # possui sequencia e porcentagem maior que threshold
        cor_resultado = (0, 0, 255) # Vermelho
    elif resultado[4] != 'OK': # possui sequencia
        cor_resultado = (0, 255, 255) # Amarelo
    else: # Tudo ok
        cor_resultado = (0, 255, 0) # Verde
    return cor_resultado

def menu_lateral(frame, status, porcentagem_falhas_atual, resultados, cor_resultado):
    cor_status=(255, 255, 255) if status=="OCIOSO" else (0, 255, 255)
    altura, largura = frame.shape[:2]
    largura_menu = 300
    cor_fundo = (50, 50, 50)
    menu = np.full((altura, largura_menu, 3), cor_fundo, dtype=np.uint8)
    margem_superior = 30
    espaco_entre_textos = 30
    cv2.putText(menu, f"Status: {status}", (10, margem_superior), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_status, 1)
    cv2.putText(menu, f"Falhas: {porcentagem_falhas_atual:.2f}%", (10, margem_superior + espaco_entre_textos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(menu, "Media | analises | Sequencia", (10, margem_superior + 3*espaco_entre_textos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for resultado in resultados:
        cor_resultado = escolhe_cor_resultado(resultado)
        multiplicador = resultados.index(resultado)+4
        cv2.putText(menu, f"{resultado[1]} | {resultado[3]} | {resultado[4]}", (10, margem_superior + multiplicador*espaco_entre_textos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_resultado, 1)
    return np.hstack((frame, menu))



while True:
    ret, frame = cap.read()
    time.sleep(0.05)
    if not ret:
        break
    roi, (rx, ry, rw, rh) = extrair_roi_corretamente(frame)  # Extrai posicao de interesse
    success, largest_rect = contornos(roi)  # Encontra maior contorno dentro do ROI estabelecido
    porcentagem_falhas_atual = 0
    if success:
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
        resultados.insert(0, resultado)
        if len(resultados) > 5: # Mantem no maximo 5 registros
            resultados.pop(5)
        frames_processados = []
    else:
        status = "OCIOSO"
    cor_resultado = ''
    frame_com_menu = menu_lateral(frame, status, porcentagem_falhas_atual, resultados, cor_resultado)
    cv2.imshow('Frame', frame_com_menu)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
