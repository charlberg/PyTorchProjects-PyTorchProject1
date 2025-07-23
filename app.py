import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_threshold = 0.8)
    model.eval()
    return model

model = load_model()
#
# def make_prediction(img):
#     img_processed = img_preprocess(img) ## (3, 500, 500)
#     prediction = model(img_processed.unsqueeze(0))[0]  ## (1, 3, 500, 500)
#     prediction["labels"] = [categories[int(label)] for label in prediction["labels"]]
#     return prediction

def make_prediction(img, score_threshold=0.8):
    img_processed = img_preprocess(img)
    outputs = model(img_processed.unsqueeze(0))[0]

    # Apply score threshold
    keep = outputs['scores'] >= score_threshold

    boxes = outputs['boxes'][keep]
    labels = outputs['labels'][keep]
    scores = outputs['scores'][keep]

    # Convert label indices to category names AFTER filtering
    label_names = [categories[int(label)] for label in labels]

    return {
        "boxes": boxes,
        "labels": label_names,
        "scores": scores
    }


def create_img_w_bboxes(img, prediction):
    img_tensor = torch.tensor(img, dtype=torch.uint8)
    img_w_bboxes = draw_bounding_boxes(img_tensor, boxes = prediction["boxes"], labels=prediction["labels"],
                                       colors=["red" if label=="person" else "green" for label in prediction["labels"]], width=2)
    img_w_bboxes_np = img_w_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_w_bboxes_np

## Dashboard
st.title("Object Detector")
upload = st.file_uploader(label = "Upload image here:", type =["png", "jpeg", "jpg"])

if upload:
    img = Image.open(upload).convert("RGB")  # ensures 3 channels
    prediction = make_prediction(img)
    img_w_bbox = create_img_w_bboxes(np.array(img).transpose(2, 0, 1), prediction)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.imshow(img_w_bbox)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]
    st.header("Predicted Probabilities")
    st.write(prediction)

