import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random

def get_loaders(path, train_transforms, test_transforms, 
                train_batch_size, test_batch_size,
                num_workers=2):

    train_data, val_data, test_data, indices = get_split(path)
    train_dataset = ESD_Dataset(train_data, train_transforms)
    val_dataset = ESD_Dataset(val_data, test_transforms)
    test_dataset = ESD_Dataset(test_data, test_transforms)


    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, 
                            num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, 
                             num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader, indices

def get_split(path, indices=None):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
   
    # data_ids = ['01', '02', '03', '04', '05', '06', '07',
    #             '08', '09', '11', '12', '13', '14', '15', '16', '17', '18', '20',
    #             '21', '22', '23', '25', '26', '27', '28', '29', '30', '31',
    #             '32', '33', '34', '35', '36']

    data = np.load(path, allow_pickle = True).item()

    data_list = []
    for _, case in data.items():
        data_list += case
    # for idx in data_ids:
    #     data_list += data[idx]

    if not indices:
        indices = list(range(len(data_list)))
        random.shuffle(indices)
    data_list = [data_list[i] for i in indices]
                     
    test_num = len(data_list) // 5
    train_data, test_data = data_list[:-test_num], data_list[-test_num:]
    # val_num = len(train_data) // 5
    # train_data, val_data = train_data[:-val_num], train_data[-val_num:]

    return train_data, test_data, test_data, indices


class ESD_Dataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.data = dataset
        self.transforms = transforms


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask, _, img_file_name = self.data[idx]
        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        label = transformed["mask"].long()
        sample = {'image': image, 'mask': label, 'id': img_file_name}
        return sample


def prepare_transforms(target_image_size=256,
                       min_intensity=0, max_intensity=1,
                       min_percentile=0, max_percentile=100,
                       perturb_test=False,
                      ):

    train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], is_check_shapes=False
    )
    test_transforms = A.Compose(
        [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()],
        is_check_shapes=False
    )
    return train_transforms,  test_transforms
    