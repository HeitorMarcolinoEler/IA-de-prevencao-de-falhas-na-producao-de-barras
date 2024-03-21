from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images/'

def image_data_url(imagem):
    retval, buffer = cv2.imencode('.png', imagem)
    if retval:
        data_url = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        return data_url
    else:
        return None

def adicionar_pontos_verdes(caminho_da_imagem):
    imagem_original = cv2.imread(caminho_da_imagem)
    pontos = [(430, 5), (585, 5), (665, 1435), (210, 1385)]  # Ponto Verdes
    for ponto in pontos:
        cv2.circle(imagem_original, ponto, 5, (0, 255, 0), -1)
    return image_data_url(imagem_original)

def processar_com_perspective(caminho_da_imagem):
    imagem_original = cv2.imread(caminho_da_imagem)
    pts1 = np.float32([(430, 5), (585, 5), (210, 1385), (665, 1435)])
    pts2 = np.float32([[0, 0], [900, 0], [0, 1600], [900, 1600]])
    unifica = cv2.getPerspectiveTransform(pts1, pts2)
    perspectiva = cv2.warpPerspective(imagem_original, unifica, (900, 1600))

    return image_data_url(pts2)

def detectar_imperfeicoes(caminho_da_imagem):
    imagem_original = cv2.imread(caminho_da_imagem)
    gray = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
    # Aplica filtro de Sobel para detecção de falhas
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    sobel = np.uint8(sobel / sobel.max() * 255)
    kernel = np.ones((3, 3), np.uint8)
    img_dilatada = cv2.dilate(sobel, kernel, iterations=1)
    return image_data_url(img_dilatada)

def processar_e_detectar_imperfeicoes(caminho_da_imagem):
    imagem_original = cv2.imread(caminho_da_imagem)
    pts1 = np.float32([(430, 5), (585, 5), (210, 1385), (665, 1435)])
    pts2 = np.float32([[0, 0], [900, 0], [0, 1600], [900, 1600]])

    matriz_transformacao = cv2.getPerspectiveTransform(pts1, pts2)
    imagem_perspectiva = cv2.warpPerspective(imagem_original, matriz_transformacao, (900, 1600))

    gray = cv2.cvtColor(imagem_perspectiva, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    sobel = np.hypot(sobelx, sobely)
    sobel = np.uint8(sobel / sobel.max() * 255)

    sobel_norm = cv2.normalize(sobel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    _, thresh = cv2.threshold(sobel_norm, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    dilatada = cv2.dilate(thresh, kernel, iterations=1)

    pixels_falhas = np.count_nonzero(dilatada)
    total_pixels = dilatada.size
    porcentagem_falhas = (pixels_falhas / total_pixels) * 100

    status = "Aprovada" if porcentagem_falhas < 0.1 else "Reprovada"
    print(f"Porcentagem de falhas detectadas: {porcentagem_falhas:.2f}% ({status})")

    return image_data_url(dilatada), status

@app.route('/', methods=['GET', 'POST'])
def exibir_imagem():
    if request.method == 'GET':
        caminho_img_original = 'images/imagem2.jpeg'
    elif request.method == 'POST':
        f = request.files['file']
        caminho_img_original = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(caminho_img_original)

    imagem_original_com_pontos_data_url = adicionar_pontos_verdes(caminho_img_original)
    imagem_processada_data_url, status = processar_e_detectar_imperfeicoes(caminho_img_original)

    if request.method == 'POST':
        os.remove(caminho_img_original)

    return render_template('index.html', imagem_original=imagem_original_com_pontos_data_url,
                           imagem_delimitada=imagem_processada_data_url, status=status)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
