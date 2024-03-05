import cv2         # Instale o pacote do OpenCV
import numpy as np # Instale o pacote do numpy

def cut_middle_object(image_path):
    # Carrega a Imagem
    image = cv2.imread(image_path)

    # Obtem as dimensões da imagem
    height, width = image.shape[:2]

    # Define a largura da seção intermediária
    middle_width = int(width * 0.75)

    # Calcula a largura das seções laterais
    side_width = int((width - middle_width) / 2)

    # Define o contorno para a seção intermediária
    contour = np.array([[[side_width, 0]], [[width - side_width, 0]],
                        [[width - side_width, height]], [[side_width, height]]], dtype=np.int32)

    # Crie uma máscara para a seção intermediária
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Aplique a máscara à imagem
    return cv2.bitwise_and(image, mask)

def filter_image(image):
    # Converta a imagem em tons de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar limite para criar uma imagem binária
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    # Encontre contornos na imagem binária
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crie uma tela para desenhar contornos
    canvas = np.zeros_like(gray_image)
    cv2.drawContours(canvas, contours, -1, (255), thickness=cv2.FILLED)

    # Extraia o objeto mais branco da imagem
    whitest_object = cv2.bitwise_and(image, image, mask=canvas)

    # Converta o objeto mais branco em tons de cinza
    gray_whitest_object = cv2.cvtColor(whitest_object, cv2.COLOR_BGR2GRAY)

    # Aplicar detecção de borda Canny
    edges = cv2.Canny(gray_whitest_object, 30, 100)

    # Encontre contornos nas bordas
    scratch_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar através de contornos de rascunho e desenhar retângulos delimitadores
    scratch_count = 0
    for contour in scratch_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x > 10 and y > 10 and x + w < image.shape[1] - 10 and y + h < image.shape[0] - 10:
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
            scratch_count += 1
    return image, scratch_count

def show_image(image, scratch_count):
    # Exibir a imagem final
    cv2.imshow('SCAN', image)
    print(f"Identificado {scratch_count} falhas de produção.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Especifique o caminho para a imagem
    image_path = 'images/imagem2.jpeg'  # Mude isso para o caminho da sua imagem

    # Processa a imagem
    image = cut_middle_object(image_path)
    image, scratch_count = filter_image(image)

    # Exibir a imagem final
    show_image(image, scratch_count)
