import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import train_transforms, SegmentationDataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True, help='Path to root dataset directory')
    parser.add_argument('--train-csv', type=str, required=True,
                        help='Path to csv, which contains names of train images and masks.')

    parser.add_argument("--epochs", default=30, type=int, help="Num epochs to train")
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size")
    parser.add_argument("--num-workers", default=6, type=int, help="Workers number for torch Dataloader")
    parser.add_argument("--lr", default=1e-3, type=float, help="Main learning rate")
    args = parser.parse_args()

    device = 'cuda'

    train_loader = DataLoader(SegmentationDataset(dataframe=pd.read_csv(args.train_csv), root_path=args.data_root,
                                                  transforms=train_transforms),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    model = smp.Unet(encoder_name="resnet34", classes=1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.BCEWithLogitsLoss()

    train_epoch_stat = {key: [] for key in ["time", "bce_loss"]}
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.

        for image, mask_gt in tqdm(train_loader, total=len(train_loader)):
            image, mask_gt = image.to(device), mask_gt.to(device)
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

    print("Total train time: {:.2f}s, mean epoch time: {:.2f}s, std: {:.2f}s".format(
        sum(train_epoch_stat['time']),
        np.mean(train_epoch_stat['time']),
        np.std(train_epoch_stat['time'])))
