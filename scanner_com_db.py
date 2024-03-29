import cv2
import numpy as np
import pymongo
from pymongo.errors import ConnectionFailure

video_path = 'static/video_esteira.mp4'
pontos_roi = [(210, 20), (250, 20), (250, 100), (210, 100)]
max_porcentagem_falhas = 0
min_porcentagem_falhas = 100
um_limiar_de_falhas = 38

def db_append_object_status(max_porcentagem_falhas, min_porcentagem_falhas, avg_porcentagem_falhas=""):
    try:
        # Base 'Produção':
        #myclient = pymongo.MongoClient("mongodb://mongouser:mongouser@vps51980.publiccloud.com.br:27017/", serverSelectionTimeoutMS=5000)
        # Base Local:
        myclient = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        myclient.admin.command('ping')
        print("Conectado com sucesso ao MongoDB.")

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
        return x.inserted_id
    except ConnectionFailure as e:
        print(f"Não foi possível conectar ao MongoDB: {e}")
    except Exception as e:
        print(f"Ocorreu um erro ao acessar o MongoDB: {e}")

if __name__ == "__main__":
    result = db_append_object_status(10, 5, 7.5)
    if result:
        print(f"Documento inserido com sucesso. ID: {result}")
    else:
        print("Documento não foi inserido.")

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

def extrair_roi_corretamente(frame):
    pts = np.array(pontos_roi, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    roi_cortada = masked[y:y+h, x:x+w]
    return roi_cortada

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

### Mexendo ainda
# Essa função deverá armazenar os valores quando entrar "EM ANALISE" para extrair os dados relevantes depois;
# deverá identificar o final do produto e depois fazer o insert dos dados relevantes no banco;
# Ver forma de extrair imagem com falha no produto caso achar;
def process_frame(frame, roi_falha, roi_falha_sobel):
    global max_porcentagem_falhas, min_porcentagem_falhas, um_limiar_de_falhas
    cor_status=(255, 255, 255)
    porcentagem_falhas_atual = np.mean(roi_falha_sobel) / 255 * 100
    status = "EM ANALISE" if porcentagem_falhas_atual < um_limiar_de_falhas else "AGUARDANDO..."
    if status == "EM ANALISE":
        max_porcentagem_falhas = max(max_porcentagem_falhas, porcentagem_falhas_atual)
        min_porcentagem_falhas = min(min_porcentagem_falhas, porcentagem_falhas_atual)
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

def main():
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
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
        frame_processado = process_frame(frame, roi_falha, roi_falha_sobel)
        cv2.imshow('Scanner Qualidade', frame_processado)
    cap.release()
    cv2.destroyAllWindows()

main()
