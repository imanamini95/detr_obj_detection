import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler

sys.path.insert(0, os.getcwd())
from scripts.config import get_train_cfg, get_val_cfg
from scripts.dataset import get_coco_dataset
from scripts.loss import mask_rcnn_loss
from scripts.models import build_model
from scripts.model import get_args_parser

from train_epoch import (
    collect_samples,
    add_train_samples,
    loss_per_epoch,
    model_checkpoint,
    train_epoch,
)


def main(
    save_folder_name,
):
    # cfg
    train_cfg = get_train_cfg(
        folder_name=save_folder_name,
    )
    val_cfg = get_val_cfg(
        folder_name=save_folder_name,
    )

    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, criterion, postprocessors = build_model(args)

    # model = torch.hub.load(
    #     "facebookresearch/detr:main", "detr_resnet50", pretrained=True
    # )

    # model.load_state_dict(
    #     torch.load(train_cfg.BEST_MODEL_PATH, map_location=train_cfg.DEVICE)
    # )

    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg.LEARNING_RATE,
        momentum=train_cfg.MOMENTUM,
        weight_decay=train_cfg.WEIGHT_DECAY,
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=train_cfg.GAMMA)

    # Load dataset
    dataset_train = get_coco_dataset(train_cfg)
    dataset_val = get_coco_dataset(val_cfg)

    # select some sequences to preview
    list_samples = collect_samples(dataset_val)
    list_samples = add_train_samples(dataset_train, list_samples)

    print(
        f"The size of training dataset is {len(dataset_train)} and the size of val dataset is {len(dataset_val)}"
    )

    train_loss_list = []
    val_loss_list = []

    for epoch in range(train_cfg.EPOCHS):
        model_checkpoint(model, epoch, list_samples, train_cfg, postprocessors)

        if epoch != 0:
            loss_per_epoch(train_loss_list, val_loss_list, epoch, train_cfg)

        train_loss, val_loss = train_epoch(
            dataset_train,
            dataset_val,
            model,
            optimizer,
            criterion,
            epoch,
            train_cfg,
            scheduler,
        )

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)


if __name__ == "__main__":
    main(
        save_folder_name="NO_PRETRAINED",
    )
