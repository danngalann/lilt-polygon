from typing import List, Dict, Union

from PIL import Image
from mmocr.apis import MMOCRInferencer
import numpy as np

class OCR:
    def __init__(self):
        self.inferencer = MMOCRInferencer(det='DBNet', rec='SAR')

    def recognize_and_detect(self, img: Image) -> List[Dict[str, Union[np.ndarray, str]]]:
        # Convert image to numpy array
        img = np.array(img)

        # Recognize and detect
        result = self.inferencer(img, show=False)

        # Convert result list to our format
        result = [
            {"polygons": entry["det_polygons"], "words": entry["rec_texts"]}
            for entry in result['predictions']
        ]

        return result


