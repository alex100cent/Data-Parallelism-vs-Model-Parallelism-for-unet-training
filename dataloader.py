import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

from torch.utils.data import Dataset

IMAGE_SIZE = (512, 512)

pre_transforms = [A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1])]

post_transforms = [A.Normalize(), ToTensorV2()]
augmentations = [
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.NoOp(),
    ], p=1),
    A.ElasticTransform(alpha=120, sigma=120 * 0.07, alpha_affine=120 * 0.03, p=.5),
    A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=15, p=.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=.5),
    A.MotionBlur(blur_limit=(3, 10), p=.5),
    A.RGBShift(r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
]

train_transforms = A.Compose([
    *pre_transforms,
    *augmentations,
    *post_transforms,
])

test_transforms = A.Compose([*pre_transforms, *post_transforms])


class SegmentationDataset(Dataset):
    def __init__(self, dataframe, root_path, transforms=None):
        self.df = dataframe
        self.root = root_path

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind: int):
        result = dict()

        result["image"] = cv2.cvtColor(cv2.imread(f"{self.root}/img/{self.df['names'].iloc[ind]}"), cv2.COLOR_BGR2RGB)
        result["mask"] = cv2.cvtColor(cv2.imread(f"{self.root}/masks_machine/{self.df['names'].iloc[ind]}"),
                                      cv2.COLOR_BGR2GRAY)
        result["mask"] = np.uint8(result["mask"] > 0)

        if self.transforms is not None:
            result = self.transforms(**result)
            result["mask"] = result["mask"].unsqueeze(0).type_as(result["image"])

        return result["image"], result["mask"]
