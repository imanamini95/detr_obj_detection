import os
import sys
import argparse

import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from src.detr_obj_detection.util import box_ops


def get_inference(frame, transforms_image, model):
    image_tensor = transforms_image(frame).unsqueeze(0).cuda()

    with torch.no_grad():
        predictions = model(image_tensor)

    out_logits, out_bbox = predictions["pred_logits"], predictions["pred_boxes"]

    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    labels = labels.squeeze(0).cpu().numpy()
    scores = scores.squeeze(0).cpu().numpy()

    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = (boxes * 512).squeeze(0).cpu().numpy()
    return scores, labels.astype("int64"), boxes.astype("int64")


def show_bbx(frame, scores, labels, boxes):
    image = frame.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.9:
            x, y, x2, y2 = box
            image = cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 2)
            image = cv2.putText(
                image,
                f"{label}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return image


if __name__ == "__main__":
    # define a video capture object
    vid = cv2.VideoCapture(0)

    model = torch.hub.load(
        "facebookresearch/detr:main", "detr_resnet50", pretrained=True
    )
    model.eval()
    model.cuda()

    transforms_image = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    while True:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        frame = cv2.resize(frame, (512, 512))
        scores, labels, boxes = get_inference(frame, transforms_image, model)
        pred = show_bbx(frame, scores, labels, boxes)

        # Display the resulting frame
        cv2.imshow("frame", pred)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
