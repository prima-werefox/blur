from typing import Optional, Callable, Tuple, Any
import pandas as pd
import os
import numpy as np

from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader


class BlurDetectionFolder(datasets.ImageFolder):
    def find_classes(self, directory: str):  # → Tuple[List[str], Dict[str, int]]
        return ['defocused_blurred',
                # 'motion_blurred',
                'sharp'], {
            'defocused_blurred': 1,
            # 'motion_blurred': 2,
            'sharp': 0,
        }


class CERTHTrainFolder(datasets.ImageFolder):
    def find_classes(self, directory: str):  # → Tuple[List[str], Dict[str, int]]
        return ['NewDigitalBlur', 'Naturally-Blurred', 'Undistorted'], {
            'NewDigitalBlur': 1,
            'Naturally-Blurred': 1,
            'Undistorted': 0,
        }


class CERTHEvalFolder(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ):
        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform
        )
        self.loader = loader
        self.samples = np.vstack([self.__read_dataset('DigitalBlurSet').to_numpy(),
                                  self.__read_dataset('NaturalBlurSet').to_numpy()])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index, :]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __read_dataset(self, ds_name: str):
        df = pd.read_csv(os.path.join(self.root, ds_name + '.csv'), header=1, sep=';', names=['filename', 'class'])
        df.dropna(inplace=True)
        df['class'] = df['class'].apply(lambda x: int(max(x, 0)))
        df['filename'] = df['filename'].apply(lambda x: os.path.join(os.path.join(self.root, ds_name),
                                                                     (x.strip() + ('' if x.strip().endswith('.jpg') else '.jpg'))))
        return df


data_transforms = {
    'train': lambda input_size: transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': lambda input_size: transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
