import argparse
import os
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import yaml
from albumentations.core.serialization import from_dict
#from iglovikov_helper_functions.config_parsing.utils import object_from_dict
#from iglovikov_helper_functions.dl.pytorch.lightning import find_average
#from iglovikov_helper_functions.dl.pytorch.utils import state_dict_from_disk
from pytorch_lightning.loggers import WandbLogger
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
from torch.utils.data import DataLoader

import pydoc

from cloths_segmentation.dataloaders import SegmentationDataset
from cloths_segmentation.metrics import binary_mean_iou
from cloths_segmentation.utils import get_samples

image_path = Path(r"/kaggle/input/segmentation-unet-imags/images")
mask_path = Path(r"/kaggle/input/segmentation-unet-labels/labels")


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-i", "--image_path", type=Path, help="Path to the config.", required=True)
    arg("-m", "--mask_path", type=Path, help="Path to the config.", required=True)
    image_path=args["image_path"]
    mask_path=args["mask_path"]
    return parser.parse_args()


os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '6080'

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def state_dict_from_disk(
    file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)

from typing import List

import torch


def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()

def object_from_dict(d, parent=None, **default_kwargs):
    print(d,'hiiiii')
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034
    print(object_type)
    x=pydoc.locate(object_type)
    print(x)
    return pydoc.locate(object_type)(**kwargs)

class SegmentPeople(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        print(hparams)
        print(type(hparams))
        self.hparams.update(hparams)

        self.model = object_from_dict(self.hparams["model"])
        if "resume_from_checkpoint" in self.hparams:
            corrections: Dict[str, str] = {"model.": ""}

            state_dict = state_dict_from_disk(
                file_path=self.hparams["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(state_dict)

        self.losses = [
            ("jaccard", 0.1, JaccardLoss(mode="binary", from_logits=True)),
            ("focal", 0.9, BinaryFocalLoss()),
        ]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(batch)

    def setup(self, stage=0):
        samples = get_samples(image_path, mask_path)

        num_train = int((1 - self.hparams["val_split"]) * len(samples))
        print(num_train)
        self.train_samples = samples[:num_train]
        self.val_samples = samples[num_train:]

        print("Len train samples = ", len(self.train_samples))
        print("Len val samples = ", len(self.val_samples))

    def train_dataloader(self):
        train_aug = from_dict(self.hparams["train_aug"])

        if "epoch_length" not in self.hparams["train_parameters"]:
            epoch_length = None
        else:
            epoch_length = self.hparams["train_parameters"]["epoch_length"]

        result = DataLoader(
            SegmentationDataset(self.train_samples, train_aug, epoch_length),
            batch_size=self.hparams["train_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams["val_aug"])
        print(self.hparams["val_aug"])
        result = DataLoader(
            SegmentationDataset(self.val_samples, val_aug, length=None),
            batch_size=self.hparams["val_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        print("Val dataloader = ", len(result))

        return result

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        total_loss = 0
        logs = {}
        for loss_name, weight, loss in self.losses:
            ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask
            logs[f"train_mask_{loss_name}"] = ls_mask

        logs["train_loss"] = total_loss

        logs["lr"] = self._get_current_lr()

        return {"loss": total_loss, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()

    def validation_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        result = {}
        for loss_name, _, loss in self.losses:
            result[f"val_mask_{loss_name}"] = loss(logits, masks)

        result["val_iou"] = binary_mean_iou(logits, masks)

        return result

    def validation_epoch_end(self, outputs):
        logs = {"epoch": self.trainer.current_epoch}

        avg_val_iou = find_average(outputs, "val_iou")

        logs["val_iou"] = avg_val_iou
        print('Avg val iou',avg_val_iou)
        return {"val_iou": avg_val_iou, "log": logs}


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    pipeline = SegmentPeople(hparams)

    Path(hparams["checkpoint_callback"]["dirpath"]).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams["trainer"],
        logger=WandbLogger(hparams["experiment_name"]),
        checkpoint_callback=object_from_dict(hparams["checkpoint_callback"]),
    )
    torch.distributed.init_process_group(backend='gloo',init_method="env://",world_size=1,rank=0)
    trainer.fit(pipeline)
    trainer.save_checkpoint("example.ckpt")


if __name__ == "__main__":
    main()
