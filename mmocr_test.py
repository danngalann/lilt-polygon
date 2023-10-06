from PIL import Image
import cv2
import numpy as np

from src.service.ocr import OCR

image = Image.open('datasets/FUNSD_polygon_augmented/dataset/testing_data/images/82092117_augmented_0.png')
ocr = OCR()

results = ocr.recognize_and_detect(image)

cv2_image = cv2.imread('datasets/FUNSD_polygon_augmented/dataset/testing_data/images/82092117_augmented_0.png')

def get_points(polygon: list):
    return np.array(polygon, dtype=np.int32).reshape((-1, 2))

for result in results:
    for polygon in result['polygons']:
        polygon = get_points(polygon)
        random_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.polylines(cv2_image, [polygon], True, random_color, 2)

cv2.imshow('image', cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()