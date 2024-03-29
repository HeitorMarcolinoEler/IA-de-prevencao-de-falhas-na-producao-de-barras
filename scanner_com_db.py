import cv2
import numpy as np
import pymongo
from pymongo.errors import ConnectionFailure
import threading
import time

video_path = 'static/video_esteira3.mp4'
pontos_roi = [(210, 40), (250, 40), (250, 440), (210, 440)]
frames_processados = []
max_porcentagem_falhas = 0
min_porcentagem_falhas = 100
um_limiar_de_falhas = 38


def db_append_object_status(min_porcentagem_falhas, max_porcentagem_falhas, avg_porcentagem_falhas):
    t = time.process_time()
    try:
        # Base 'Produção':
        myclient = pymongo.MongoClient("mongodb://mongouser:mongouser@vps51980.publiccloud.com.br:27017/", serverSelectionTimeoutMS=5000)
        # Base Local:
        # myclient = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        
        db = myclient["db_esteira"]
        collection_rodape = db["objeto_rodape"]
        rodape_specs = {
            "nome": "rodape1",
            "max_porcentagem_falhas": max_porcentagem_falhas,
            "min_porcentagem_falhas": min_porcentagem_falhas,
            "med_porcentagem_falhas": avg_porcentagem_falhas,
            "caminho_frame_processado": "images/qualquer.png"
        }
        x = collection_rodape.insert_one(rodape_specs)
        tempo = time.process_time() - t
        print(f"FEITO INSERT EM {tempo:.3f}ms")
        return x.inserted_id
    except ConnectionFailure as e:
        print(f"Não foi possível conectar ao MongoDB: {e}")
    except Exception as e:
        print(f"Ocorreu um erro ao acessar o MongoDB: {e}")

def extrair_roi_corretamente(frame):
    pts = np.array(pontos_roi, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    roi_cortada = masked[y:y+h, x:x+w]
    return roi_cortada

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

def menu_lateral(frame, status, cor_status, porcentagem_falhas, resultado, cor_resultado, roi_falha=None, roi_falha_sobel=None):
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

def process_frame(frame, roi_falha, roi_falha_sobel):
    global max_porcentagem_falhas, min_porcentagem_falhas, um_limiar_de_falhas, frames_processados
    cor_status=(255, 255, 255)
    frames_processados_final = []
    porcentagem_falhas_atual = np.mean(roi_falha_sobel) / 255 * 100
    status = "EM ANALISE" if porcentagem_falhas_atual < um_limiar_de_falhas else "AGUARDANDO..."
    if status == "EM ANALISE":
        frames_processados.append(porcentagem_falhas_atual)
        max_porcentagem_falhas = max(max_porcentagem_falhas, porcentagem_falhas_atual)
        min_porcentagem_falhas = min(min_porcentagem_falhas, porcentagem_falhas_atual)
    else:
        if len(frames_processados) != 0:
            frames_processados_final = frames_processados # Guarda todos os frames processados
            frames_processados = []
        else:
            frames_processados = []
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
    return frame_com_menu, frames_processados_final

def analise_de_frames(frames_processados_final):
    mini = min(frames_processados_final)
    maxi = max(frames_processados_final)
    avg = sum(frames_processados_final) / len(frames_processados_final)
    return f"{mini:.2f}%", f"{maxi:.2f}%", f"{avg:.2f}%"


def main():
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        # time.sleep(0.1) # delay pra debugar
        ret, frame = cap.read()
        if not ret: # Quando acabar o video, reiniciar (apertando R) ou fechar depois de 7 segundos.
            key = cv2.waitKey(7000) & 0xFF
            if key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Sair
            break
        roi_falha = extrair_roi_corretamente(frame)
        roi_falha_sobel = area_de_interesse(frame)
        frame_processado, frames_processados_final = process_frame(frame, roi_falha, roi_falha_sobel)
        if len(frames_processados_final) != 0:
            mini, maxi, avg = analise_de_frames(frames_processados_final)
            x = threading.Thread(target=db_append_object_status, args=(mini, maxi, avg)) # Cria processo
            x.start() # começa
            # db_append_object_status(mini, maxi, avg) # Processo unico sincrono
            print(f"Final com {len(frames_processados_final)} frames processados, maximo {maxi} minimo {mini} media {avg}")
        cv2.imshow('Scanner Qualidade', frame_processado)
    cap.release()
    cv2.destroyAllWindows()


main()
