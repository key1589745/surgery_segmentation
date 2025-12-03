import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
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

def get_split(path):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    dataset = np.load(path, allow_pickle=True).item()

    if 'image' in dataset:
        # If the dataset is already flattened, we cannot recover video structure easily
        # Assuming standard ESD format where it is a dict of videos
        print("Warning: Dataset seems already flattened. Continuous testing might not work as expected.")
        total = len(dataset['image'])
        indices = list(range(total))
        random.shuffle(indices)
        dataset = {'image': [dataset['image'][i] for i in indices], 'mask': [dataset['mask'][i] for i in indices]}
        
        val_len = total // 5
        test_len = total // 5
        
        train = {'image':dataset['image'][:-(val_len+test_len)], 'mask':dataset['mask'][:-(val_len+test_len)]}
        vali = {'image':dataset['image'][-(val_len+test_len):-test_len], 'mask':dataset['mask'][-(val_len+test_len):-test_len]}
        test = {'image':dataset['image'][-test_len:], 'mask':dataset['mask'][-test_len:]}
        return train, vali, test, indices

    # Split by video ID to ensure continuous test data
    video_keys = sorted(list(dataset.keys()))
    random.shuffle(video_keys)
    
    total_videos = len(video_keys)
    val_len = total_videos // 5
    test_len = total_videos // 5
    
    # Split videos
    test_vids = video_keys[:test_len]
    val_vids = video_keys[test_len:test_len+val_len]
    train_vids = video_keys[test_len+val_len:]
    
    def collect_data(vids):
        data = {'image': [], 'mask': [], 'video_id': [], 'frame_idx': []}
        for vid in vids:
            frames = dataset[vid]
            for idx, item in enumerate(frames):
                data['image'].append(item[0])
                data['mask'].append(item[1])
                data['video_id'].append(vid)
                data['frame_idx'].append(idx)
        return data

    train = collect_data(train_vids)
    vali = collect_data(val_vids)
    test = collect_data(test_vids)
    
    # Indices not meaningful here as we reconstructed the lists
    indices = [] 
    
    return train, vali, test, indices


class ESD_Dataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.data = dataset
        self.transforms = transforms


    def __len__(self):
        return len(self.data['image'])

    def __getitem__(self, idx):
        image, mask = self.data['image'][idx], self.data['mask'][idx]
        if len(image.shape) < 4:
            image = np.expand_dims(image,0).repeat(8,0)
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        image = tv_tensors.Video(image.transpose(0,3,1,2))
        mask = tv_tensors.Mask(mask).long()
        image, mask = self.transforms(image, mask)

        sample = {'image': image, 'mask': mask}
        if 'video_id' in self.data:
            sample['video_id'] = self.data['video_id'][idx]
        if 'frame_idx' in self.data:
            sample['frame_idx'] = self.data['frame_idx'][idx]

        return sample



def prepare_transforms():

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(0.5),
        v2.RandomAffine(degrees=15,translate=(0.1,0.1),scale=(0.9,1.1)),
        v2.ColorJitter(contrast=0.5,brightness=0.5,saturation=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    test_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                 )
    return train_transforms,  test_transforms


class ESD_DataLoaders:
    def __init__(self, path, transforms, train_batch_size, test_batch_size, num_workers=2, num_classes=None, dataset_name="ESD"):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        train_transforms, test_transforms = transforms
        
        train_loader, val_loader, test_loader, indices = get_loaders(
            path, train_transforms, test_transforms,
            train_batch_size, test_batch_size, num_workers
        )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader



    