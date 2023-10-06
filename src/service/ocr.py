from typing import List, Dict, Union

from PIL import Image
from mmocr.apis import MMOCRInferencer
import numpy as np


class OCR:
    def __init__(self, min_distance=50):
        self.inferencer = MMOCRInferencer(det='DBNet', rec='SAR')
        self.min_distance = min_distance

    def compute_distance(self, poly1: List[float], poly2: List[float]) -> float:
        """
        Compute the horizontal distance between two polygons.
        """
        xright_poly1 = poly1[2]
        xleft_poly2 = poly2[0]

        return abs(xleft_poly2 - xright_poly1)

    def merge_two_polygons(self, polygon1: List[float], polygon2: List[float]) -> List[float]:
        """
        Merge two polygons by combining the corners to keep the tilt of the text.
        """
        xleft_top = polygon1[0]
        yleft_top = polygon1[1]

        xright_top = polygon2[2]
        yright_top = polygon2[3]

        xright_bottom = polygon2[4]
        yright_bottom = polygon2[5]

        xleft_bottom = polygon1[6]
        yleft_bottom = polygon1[7]

        return [xleft_top, yleft_top, xright_top, yright_top, xright_bottom, yright_bottom, xleft_bottom, yleft_bottom]

    def group_polygons_into_sentences(self, result) -> List[
        Dict[str, Union[np.ndarray, str]]]:
        """
        Group polygons and words into sentences based on distance criteria.
        """
        grouped_results = []

        for entry in result:
            polygons = entry["polygons"]
            words = entry["words"]

            current_group = {"polygons": [], "words": []}

            for i in range(len(polygons)):
                polygon = polygons[i]
                word = words[i]

                if len(current_group["polygons"]) == 0:
                    current_group["polygons"].append(polygon)
                    current_group["words"] = word
                    continue

                last_polygon = current_group["polygons"][-1]

                distance = self.compute_distance(last_polygon, polygon)
                if distance < self.min_distance:
                    merged_polygon = self.merge_two_polygons(last_polygon, polygon)
                    current_group["polygons"] = [merged_polygon]
                    current_group["words"] += " " + word
                else:
                    grouped_results.append(current_group)
                    current_group = {"polygons": [polygon], "words": word}

            grouped_results.append(current_group)

        return grouped_results

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

        # Group polygons into sentences
        grouped_results = self.group_polygons_into_sentences(result)

        return grouped_results
