import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import train_transforms, SegmentationDataset

DEVICE_0 = "cuda:0"
DEVICE_1 = "cuda:1"


class ModelParallelUnet(smp.Unet):
    def __init__(self, split_size, *args, **kwargs):
        super(ModelParallelUnet, self).__init__(*args, **kwargs)
        self.split_size = split_size

        # Load 1 part to DEVICE_0
        self.encoder = self.encoder.to(DEVICE_0)

        # Load 2 part to DEVICE_1
        self.encoder.layer4 = self.encoder.layer4.to(DEVICE_1)
        self.decoder = self.decoder.to(DEVICE_1)
        self.segmentation_head = self.segmentation_head.to(DEVICE_1)

        # split encoder into two parts
        self.encoder_stages_0 = [
            nn.Identity(),
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
        ]
        self.encoder_stages_1 = [
            self.encoder.layer4,
        ]

    def forward_0(self, x):
        features = []
        for i in range(len(self.encoder_stages_0)):
            x = self.encoder_stages_0[i](x)
            features.append(x)
        features = [f.to(DEVICE_1) for f in features]

        return features

    def forward_1(self, features):
        x = features[-1]
        for i in range(len(self.encoder_stages_1)):
            x = self.encoder_stages_1[i](x)
            features.append(x)
        return self.segmentation_head(self.decoder(*features))

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.forward_0(s_next)
        ret = []

        for s_next in splits:
            masks = self.forward_1(s_prev)
            ret.append(masks)

            s_prev = self.forward_0(s_next)

        masks = self.forward_1(s_prev)
        ret.append(masks)

        return torch.cat(ret)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--split-size', type=int, default=2,
                        help='Batch split size. This parameter affects the performance')
    parser.add_argument('--data-root', type=str, required=True, help='Path to root dataset directory')
    parser.add_argument('--train-csv', type=str, required=True,
                        help='Path to csv, which contains names of train images and masks.')

    parser.add_argument("--epochs", default=30, type=int, help="Num epochs to train")
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size")
    parser.add_argument("--num-workers", default=5, type=int, help="Workers number for torch Dataloader")
    parser.add_argument("--lr", default=1e-3, type=float, help="Main learning rate")
    args = parser.parse_args()

    train_loader = DataLoader(SegmentationDataset(dataframe=pd.read_csv(args.train_csv), root_path=args.data_root,
                                                  transforms=train_transforms),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    model = ModelParallelUnet(split_size=args.split_size, encoder_name="resnet34", classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_epoch_stat = {key: [] for key in ["time", "bce_loss"]}
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.

        for image, mask_gt in tqdm(train_loader, total=len(train_loader)):
            image, mask_gt = image.to(DEVICE_0), mask_gt.to(DEVICE_1)
            pred = model(image)

            loss = criterion(pred, mask_gt)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() / len(train_loader)

        train_epoch_stat["time"].append(time.time() - epoch_start)
        train_epoch_stat["bce_loss"].append(epoch_loss)
        print("{}/{}, epoch time: {:.2f}s, bce_loss: {:.4f}".format(epoch + 1, args.epochs,
                                                                    train_epoch_stat["time"][-1],
                                                                    train_epoch_stat["bce_loss"][-1]))

    print("split_size: {}, Total train time: {:.2f}s, mean epoch time: {:.2f}s, std: {:.2f}s".format(
        args.split_size,
        sum(train_epoch_stat['time']),
        np.mean(train_epoch_stat['time']),
        np.std(train_epoch_stat['time'])))
