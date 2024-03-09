import cv2
import numpy as np

image = cv2.imread('images/imagem8.png', 0) # Load the image
blurred = cv2.GaussianBlur(image, (5, 5), 0) # Apply GaussianBlur to reduce noise and improve contour detection
_, thresh = cv2.threshold(blurred, 185, 255, cv2.THRESH_BINARY_INV) # Use adaptive thresholding
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours
for contour in contours: # Draw contours on the original image
    if cv2.contourArea(contour) > 40:  # Adjust this value based on your requirement
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow('Detected Paint Issues', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
