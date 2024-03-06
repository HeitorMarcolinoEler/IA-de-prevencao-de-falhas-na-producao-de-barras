import cv2
import numpy as np
import base64
import os
from flask import Flask, render_template, request


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images/'


@app.route('/', methods = ['GET', 'POST'])
def exibir_imagem_processada():
    if request.method == 'GET': # Imagem Default
        imagem_processada = processar_imagem('images/imagem2.jpeg') # Especifique o caminho para a imagem
    elif request.method == 'POST':
        f = request.files['file']
        caminho_img_temp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(caminho_img_temp)
        imagem_processada = processar_imagem(caminho_img_temp)
        os.remove(caminho_img_temp)
    return render_template('index.html', imagem_processada=imagem_processada) # Processa a imagem


def processar_imagem(caminho_da_imagem):
    def cut_middle_object(image_path):
        image = cv2.imread(image_path) # Carrega a Imagem
        height, width = image.shape[:2] # Obtem as dimensões da imagem
        middle_width = int(width * 0.75) # Define a largura da seção intermediária
        side_width = int((width - middle_width) / 2) # Calcula a largura das seções laterais
        contour = np.array([[[side_width, 0]], [[width - side_width, 0]],[[width - side_width, height]], [[side_width, height]]], dtype=np.int32) # Define o contorno para a seção intermediária
        mask = np.zeros_like(image) # Crie uma máscara para a seção intermediária
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        return cv2.bitwise_and(image, mask) # Aplica a máscara à imagem

    def filter_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converta a imagem em tons de cinza
        _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY) # Aplicar limite para criar uma imagem binária
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontra contornos na imagem binária
        canvas = np.zeros_like(gray_image)
        cv2.drawContours(canvas, contours, -1, (255), thickness=cv2.FILLED) # Cria uma tela e desenha os contornos
        whitest_object = cv2.bitwise_and(image, image, mask=canvas) # Extraia o objeto mais branco da imagem
        gray_whitest_object = cv2.cvtColor(whitest_object, cv2.COLOR_BGR2GRAY) # Converte o objeto mais branco em tons de cinza
        edges = cv2.Canny(gray_whitest_object, 30, 100) # Aplica detecção de borda
        scratch_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontra contornos nas bordas

        # Iterar através de contornos de rascunho e desenha retângulos delimitadores
        scratch_count = 0
        for contour in scratch_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x > 10 and y > 10 and x + w < image.shape[1] - 10 and y + h < image.shape[0] - 10:
                cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
                scratch_count += 1
        return image, scratch_count

    image = cut_middle_object(caminho_da_imagem) # Processa a imagem
    image, scratch_count = filter_image(image)

    retval, buffer = cv2.imencode('.png', image) # Converte a imagem processada para uma URL de dados
    imagem_processada_data_url = f'data:image/png;base64,{base64.b64encode(buffer).decode("utf-8")}'
    return imagem_processada_data_url # Retorna a imagem processada

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0:5000')
