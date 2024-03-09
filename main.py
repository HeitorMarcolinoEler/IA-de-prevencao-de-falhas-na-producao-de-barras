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
        imagem_processada = processar_imagem('images/imagem4_ajustada.jpeg') # Especifique o caminho para a imagem
    elif request.method == 'POST':
        f = request.files['file']
        caminho_img_temp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(caminho_img_temp)
        imagem_processada = processar_imagem(caminho_img_temp)
        os.remove(caminho_img_temp)
    return render_template('index.html', imagem_processada=imagem_processada) # Processa a imagem


def processar_imagem(caminho_da_imagem):
    def image_data_url(image):
        retval, buffer = cv2.imencode('.png', image) # Converte a imagem processada para uma URL de dados
        imagem_processada_data_url = f'data:image/png;base64,{base64.b64encode(buffer).decode("utf-8")}'
        return imagem_processada_data_url
    
    def cut_middle_object(image):
        height, width = image.shape[:2] # Obtem as dimensões da imagem
        middle_width = int(width * 1) # Define a largura a ser cortada (0 a 1)
        side_width = int((width - middle_width) / 2) # Calcula a largura das seções laterais
        curvatura_do_objeto = 225 # Ponto superior a ser cortado
        contour = np.array([[[curvatura_do_objeto, 0]], [[width - curvatura_do_objeto, 0]],[[width - side_width, height]], [[side_width, height]]], dtype=np.int32) # Define o contorno para a seção intermediária
        mask = np.zeros_like(image) # Crie uma máscara para a seção intermediária
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image # Aplica a máscara à imagem


    def gray_binary_contours(image, threshold_min, threshold_max):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converta a imagem em tons de cinza
        _, binary_image = cv2.threshold(gray_image, threshold_min, threshold_max, cv2.THRESH_BINARY) # Ajustar threshold para filtrar branco(min) e preto(max)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontra contornos
        return gray_image, binary_image, contours
    
    def find_scratches(image):
        gray_image, binary_image, contours = gray_binary_contours(image, 195, 255)
        canvas = np.zeros_like(gray_image)
        cv2.drawContours(canvas, contours, -1, (255), thickness=cv2.FILLED) # Cria uma tela e desenha os contornos
        whitest_object = cv2.bitwise_and(image, image, mask=canvas) # Extraia o objeto mais branco da imagem
        gray_whitest_object = cv2.cvtColor(whitest_object, cv2.COLOR_BGR2GRAY) # Converte o objeto mais branco em tons de cinza
        edges = cv2.Canny(gray_whitest_object, 50, 150) # Aplica detecção de borda
        scratch_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontra contornos nas bordas
        scratch_count = 0
        for contour in scratch_contours: # Iterar através de contornos de rascunho e desenha retângulos delimitadores
            x, y, w, h = cv2.boundingRect(contour)
            if x > 10 and y > 10 and x + w < image.shape[1] - 10 and y + h < image.shape[0] - 10:
                cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
                scratch_count += 1
        print(f"Arranhados: {scratch_count}")
        return image

    def find_paint_issues(image): # Irá filtrar tons de branco e desenhar
        gray_image, binary_image, contours = gray_binary_contours(image, 190, 255) # Regular
        paint_issues_count = 0
        for contour in contours:
            if cv2.contourArea(contour) > 30 and cv2.contourArea(contour) < 2000: # Ajustar tamanho do contorno
                x, y, w, h = cv2.boundingRect(contour) # Desenha contornos na imagem recebida
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                paint_issues_count += 1
        print(f"Problemas de pintura: {paint_issues_count}")
        # cv2.imshow('result', binary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image
    

    def main_proccess(): # Chamada de ferramental para análise da imagem
        image = cv2.imread(caminho_da_imagem)
        image = cut_middle_object(image)
        image = find_scratches(image)
        image = find_paint_issues(image)
        return image
    

    image = main_proccess()
    imagem_processada_data_url = image_data_url(image)
    return imagem_processada_data_url # Retorna a imagem processada

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
