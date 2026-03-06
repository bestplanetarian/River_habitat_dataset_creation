"""Pascal VOC Semantic Segmentation Dataset."""
import os
import torch
import numpy as np
import csv 

from PIL import Image
from .segbase import SegmentationDataset


class SubstrateSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """


    #'/home/swz45/Documents/Sasha_modified/Cropped_img/train/Substrate_shadow_model/Unfiltered_task/agreed_approach'

    #'/home/swz45/Documents/cascade_task/output_split/test'

    #BASE_DIR = 'VOC2012'
    NUM_CLASS = 5

    def __init__(self, root='../datasets/voc', split='train', mode=None, transform=None, **kwargs):
        super(SubstrateSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        #Change the directory here
        _substrate_root = '../test_images'

        

        _mask_dir = os.path.join(_substrate_root, 'label_removed_labels')
        _image_dir = os.path.join(_substrate_root, 'label_removed_images')
        # train/val/test splits are pre-cut
        substrate_root = '../text_file'
        _splits_dir = os.path.join(substrate_root, 'spliting')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'valid.txt')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'valid.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        self._classes = [0, 1, 2, 3, 4]
        self.class_labels = self._precompute_class_labels()

        
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir,  line.rstrip('\n'))
                name = line.rstrip('\n')
                assert os.path.isfile(_image)
                self.images.append(_image)
                if split != 'test':
                    #line.strip('\n').split('.')[]
                    #parts = line.rstrip('\n').split('_')  # e.g., ["one", "two", "three", "four"]

                    # Combine everything except the last element
                    #combined = '_'.join(parts[:-1]) + '_binarymask.png'
                    _mask = os.path.join(_mask_dir, name)
                    #print(_mask)
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)
                else:
                    _mask = os.path.join(_mask_dir, name)
                    #print(_mask)
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)
        
        '''
        with open(os.path.join(_split_f), 'r', newline='') as csvfile:
             reader = csv.reader(csvfile)
             for row in reader:
            # row[0] = image filename, row[1] = mask filename
            # e.g.: row might look like ["image001.jpg", "image001_binarymask.png"]
                 #print(row)
                 image_name = row[0]
                 #mask_name = row[1]  # only needed if 'split' != 'test'

            # Build full paths
                 image_path = os.path.join(_image_dir, image_name)

            # Make sure they exist
                 assert os.path.isfile(image_path), f"Image file not found: {image_path}"
                 self.images.append(image_path)
                 #mask_path = os.path.join(_mask_dir, mask_name)
                 #self.masks.append(mask_path)

                 if split != 'test':
                    mask_path = os.path.join(_mask_dir, image_name)
                    assert os.path.isfile(mask_path), f"Mask file not found: {mask_path}"
                    self.masks.append(mask_path)
                 else:
                    mask_path = os.path.join(_mask_dir, image_name)
                    self.masks.append(mask_path)
        '''

        #if split != 'test':
        #    assert (len(self.images) == len(self.masks))
        self.class_labels = self._precompute_class_labels()
        print('Found {} images in the folder {}'.format(len(self.images), _substrate_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask, os.path.basename(self.images[index])
    
    def _precompute_class_labels(self):
        labels = []
        print('The mask is:', self.masks)
        for mask_path in self.masks:
            mask = Image.open(mask_path)
            mask = self._mask_transform(mask)  # Convert to tensor first
            if (mask == 1).any():  # If shadow exists
                labels.append(1)
            elif (mask == 2).any():  # If others exist
                labels.append(2)
            elif (mask == 3).any():  # If others exist
                labels.append(3)
            elif (mask == 4).any():  # If others exist
                labels.append(4)
            else:
                labels.append(0)  # Background-only image
        return labels

    def get_class_labels(self):
        return self.class_labels

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        #target = np.array(mask).astype('int32')
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        #return ('back', 'Bedrock','Fine', 'Woody', 'Boulder','Rocky_Fine')
        #return ('Background', 'Bedrock','Shadow','Fine','Bank','Woody','Boulder','Rocky Fine')
        #return ('back', 'Shadow', 'Bank', 'Otherclass')
        #return ('Back', 'Fine', 'Boulder', 'RockyFine')
        return ('back', 'shadow', 'fine', 'boulder', 'rocky fine')
  
if __name__ == '__main__':
    dataset = SubstrateSegmentation()
