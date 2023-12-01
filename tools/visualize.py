import numpy as np
import cv2

def visualize(image: np.ndarray,save_path: str,points: np.ndarray) -> None:
    vis_image = image.copy()
    for x,y in points[0]:
        vis_image = cv2.circle(vis_image,(int(x),int(y)),1,(0,0,255),-1)

    cv2.imwrite(save_path, vis_image)