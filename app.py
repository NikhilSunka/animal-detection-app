import os
os.system("pip install 'git+https://github.com/facebookresearch/detectron2.git'")

import torch
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from huggingface_hub import hf_hub_download

# --- Load Models from Hugging Face Hub ---

# YOLOv8m
yolo_path = hf_hub_download(repo_id="nikhilsunka/animal-detection-models", filename="best.pt")
yolo_model = YOLO(yolo_path)

# Faster R-CNN
fasterrcnn_path = hf_hub_download(repo_id="nikhilsunka/animal-detection-models", filename="model_final.pth")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 50
cfg.MODEL.WEIGHTS = fasterrcnn_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

# --- Class Names ---
classes = [
    "Bat_(Animal)", "Bear", "Bird", "Brown_bear", "Camel", "Cat", "Caterpillar", "Cattle", "Cheetah",
    "Chicken", "Crab", "Crocodile", "Deer", "Dog", "Duck", "Eagle", "Elephant", "Fish", "Fox", "Frog",
    "Giraffe", "Goat", "Goldfish", "Harbor_seal", "Horse", "Insect", "Jellyfish", "Kangaroo", "Lion",
    "Lizard", "Lobster", "Monkey", "Ostrich", "Panda", "Parrot", "Pig", "Polar_bear", "Rabbit", "Raccoon",
    "Red_panda", "Sea_lion", "Sea_turtle", "Seahorse", "Sheep", "Snake", "Squirrel", "Starfish", "Tiger",
    "Turtle", "Zebra"
]
MetadataCatalog.get("custom_animals").thing_classes = classes

# --- Inference Function ---
def detect(model_choice, image):
    if model_choice == "YOLOv8m":
        results = yolo_model.predict(image)
        res_plotted = results[0].plot()
        return Image.fromarray(res_plotted)
    elif model_choice == "Faster R-CNN":
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        outputs = predictor(img_bgr)
        v = Visualizer(img_np[:, :, ::-1], metadata=MetadataCatalog.get("custom_animals"), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return Image.fromarray(out.get_image())

# --- Gradio UI ---
interface = gr.Interface(
    fn=detect,
    inputs=[
        gr.Radio(["YOLOv8m", "Faster R-CNN"], label="Choose Detection Model"),
        gr.Image(type="pil", label="Upload an Animal Image")
    ],
    outputs="image",
    title="Animal Detection Comparison App",
    description="Upload an animal image and compare predictions between YOLOv8m and Faster R-CNN."
)

interface.launch()
