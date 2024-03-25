import cv2
import io
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

### initialize model
model_path = os.path.join("assets","models","cctv-feed-ml_v3-post-prd-022024_yolov8l.pt")
model = YOLO(model_path)


def __read_img_from_bytes(byte_img): # function byte_img to img
    image = Image.open(io.BytesIO(byte_img))
    image_np = np.array(image)

    return image_np


def ai_agent(byte_image): # Function run ai
    ### convert byte_img to img
    img = __read_img_from_bytes(byte_image)

    ### run model ---> Paste you code below <---- 
    results = model.predict(img)

    ret = []
    for output in results:
        output = output.cpu()

        ret.append({
                        "class": output.boxes.cls.to(int).tolist(),
                        "conf": output.boxes.conf.tolist(),
                        "bdbox": output.boxes.xyxy.to(int).tolist()
                    })


    return ret
