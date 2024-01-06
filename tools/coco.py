from abc import abstractmethod
import json
import pathlib
import shutil
import cv2
import numpy as np
from tqdm import tqdm


class COCO:
    """
    将数据转换为COCO格式，并保存至文件中
    """
    def __init__(self,coco_root) -> None:
        self.ann_id = -1
        self.image_id = -1
        if isinstance(coco_root,str):
            coco_root = pathlib.Path(coco_root)
        coco_root.mkdir(exist_ok=True,parents=True)
        self.coco_root = coco_root
        self.images_metas = []
        self.image_anns = []

    def to_coco(self):
        """
        保存数据到文件中
        """
        instance = dict(
                info = 'spytensor created',
                license = ['license'],
                images = self.images_metas,
                annotations = self.image_anns,
                categories = self.get_categories()  # 需要在子类中写明
            )
        self.write_to_json(instance)
        return instance


    def write_to_json(self,instance):
        coco_json = self.coco_root /'annotations'/ 'instances2017.json'
        coco_json.parent.mkdir(exist_ok=True,parents=True)
        with open(coco_json, 'w', encoding='utf-8') as f:
            json.dump(instance,f, ensure_ascii=False, indent=1)
        print(coco_json,": 写入成功！！")

    @staticmethod
    def get_categories():
        return [
            {
                "supercategory": "prawn",
                "id": 1,
                "name": "penaeus monodon",
                "keypoints": [
                    "abdomen"
                ],
                "skeleton": [
                ]
            }]

    def build_meta(self,image_path):
        self.image_id += 1
        image =cv2.imread(str(image_path))
        h,w,_ = image.shape
        image = dict(
            height = h,
            width = w,
            id = self.image_id,
            file_name = image_path.name
        )
        return image

    def get_coco_keypoints(self,points):
        keypoints = []
        for x,y in points:
            keypoints.append(x)
            keypoints.append(y)
            keypoints.append(2)
        return keypoints

    def build_anno(self,points):
        """ 返回单个点的标注信息
            points格式应为[[x0,y0],[x1,y1],[x2,y2]]"""
        self.ann_id += 1
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.image_id
        annotation['category_id'] = 1 
        annotation['segmentation'] = [points[0][0],points[0][1],50,50]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        annotation['num_keypoints'] = len(points)

        annotation['keypoints'] = self.get_coco_keypoints(points)
        return annotation
    
    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, 100, 100]
    def convert(self,points,image):
        
        self.images_metas.append(self.build_meta(image))
        for k in range(len(points)):
            for x,y in points[k]:
                self.image_anns.append(self.build_anno([[x,y]]))
