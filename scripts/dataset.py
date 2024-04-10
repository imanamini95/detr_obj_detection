import os
import json
import urllib.request

from pycocotools import mask as coco_mask
import numpy as np
import cv2
import torch


def get_coco_dataset(cfg):
    return COCODataset(cfg)


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = cfg.DATA_DIRECTORY
        self.transform_image = cfg.transforms_image
        self.debug_ = cfg.DEBUG_TRAIN
        self.target_shape = cfg.SHAPE
        self.train = cfg.TRAIN

        with open(self.root_dir, "r") as f:
            self.data = json.load(f)

    def get_categories(self, items):
        categories = torch.zeros((len(items)), dtype=torch.int64)
        for i, item in enumerate(items):
            category = self.data["annotations"][item]["category_id"]
            categories[i] = category

        return categories

    def correct_missed_annotation(self, bbx):
        if bbx[2] <= 0:
            bbx[2] = 1
        if bbx[3] <= 0:
            bbx[3] = 1
        return bbx

    def resize_bbx(self, bbx):
        x, y, w, h = bbx
        old_height, old_width = self.shape[0], self.shape[1]
        new_height, new_width = self.target_shape

        scale_w = new_width / old_width
        scale_h = new_height / old_height

        x_new = x * scale_w
        y_new = y * scale_h
        w_new = w * scale_w
        h_new = h * scale_h

        return x_new, y_new, w_new, h_new

    def get_bbxs(self, items):
        bbxs = np.zeros((len(items), 4))
        for i, item in enumerate(items):
            bbx = self.data["annotations"][item]["bbox"]
            bbx = self.correct_missed_annotation(bbx)
            bbxs[i, :] = (bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3])

        if self.debug_:
            bbx_mask = np.zeros((self.target_shape))
            bbx = bbxs[0, :]
            x1, y1, x2, y2 = bbx
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            cv2.rectangle(bbx_mask, (x1, y1), (x2, y2), 255, thickness=2)
            cv2.imwrite("./.debug/bbx.png", bbx_mask)

        bbxs = torch.tensor(bbxs).float()
        return bbxs

    def get_masks(self, items):
        seg_masks = np.zeros(
            (len(items), self.target_shape[0], self.target_shape[1])
        ).astype("uint8")

        for i, item in enumerate(items):
            annotation = self.data["annotations"][item]
            orig_mask = np.zeros((self.shape[0], self.shape[1]), dtype="uint8")
            if annotation["iscrowd"] == 1:
                compressed_rle = coco_mask.frPyObjects(
                    annotation["segmentation"],
                    annotation["segmentation"].get("size")[0],
                    annotation["segmentation"].get("size")[1],
                )
                orig_mask = coco_mask.decode(compressed_rle)
                seg_masks[i] = cv2.resize(orig_mask, self.target_shape)
                continue

            seg_pts = annotation["segmentation"][0]
            contour = np.array(
                [(seg_pts[i], seg_pts[i + 1]) for i in range(0, len(seg_pts), 2)]
            )
            contour = np.expand_dims(contour, axis=1).astype("int64")

            cv2.drawContours(
                orig_mask,
                [contour],
                contourIdx=0,
                color=255,
                thickness=cv2.FILLED,
            )
            seg_masks[i] = cv2.resize(orig_mask, self.target_shape)

        if self.debug_ and len(items) > 0:
            cv2.imwrite("./.debug/masks.png", seg_masks[0])

        seg_masks = torch.tensor(seg_masks) / 255.0
        return seg_masks

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        # get the image
        while True:
            try:
                img_url = self.data["images"][idx]["coco_url"]
                try:
                    req = urllib.request.urlopen(img_url)
                except:
                    img_url = self.data["images"][idx]["flickr_url"]
                    req = urllib.request.urlopen(img_url)

                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)
                break
            except:
                pass

        self.shape = image.shape

        image = cv2.resize(image, (self.target_shape))

        if len(image.shape) == 2:
            image = np.dstack([image, image, image])

        if self.debug_:
            cv2.imwrite("./.debug/image.png", image)

        if self.transform_image:
            image = self.transform_image(image)

        # get the ground truth
        interest_objects_idx = []
        for i in range(len(self.data["annotations"])):
            if (
                self.data["annotations"][i]["image_id"]
                == self.data["images"][idx]["id"]
            ):
                interest_objects_idx.append(i)

        categories = self.get_categories(interest_objects_idx)
        bbxs = self.get_bbxs(interest_objects_idx)
        seg_masks = self.get_masks(interest_objects_idx)

        grount_truth = {"labels": categories, "boxes": bbxs, "masks": seg_masks}

        return image, grount_truth


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.getcwd())
    from scripts.config import get_val_cfg

    val_cfg = get_val_cfg()

    dataset = get_coco_dataset(val_cfg)

    x, y = dataset[1000]
