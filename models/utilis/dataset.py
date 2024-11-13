import cv2 as cv
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from typing import Callable
from sklearn.model_selection import train_test_split
class TurtlesDataset(Dataset):
    def __init__(self, coco: COCO, image_ids:list, 
                 transform: Callable[[np.ndarray, dict], dict] | None = None,
                 dataset_path: str = "data",
                 resize: tuple[int, int] = (512, 512)):
        self.coco = coco
        self.image_ids = image_ids
        self.transform = transform
        self.catIds = coco.getCatIds()
        self.dataset_path = dataset_path
        self.resize = resize

    def __len__(self):
        return len(self.image_ids)

    def _process_mask(self, image_id: int, image: np.ndarray) -> np.ndarray:
        turtle_mask = self.coco.getAnnIds(imgIds=image_id, catIds=1, iscrowd=None)  
        leg_mask = self.coco.getAnnIds(imgIds=image_id, catIds=2, iscrowd=None)  
        head_mask = self.coco.getAnnIds(imgIds=image_id, catIds=3, iscrowd=None)  
        turtle_mask = self.coco.loadAnns(turtle_mask)
        leg_mask = self.coco.loadAnns(leg_mask)
        head_mask = self.coco.loadAnns(head_mask)
        mask1 = np.zeros((image.shape[0], image.shape[1]))
        for i in range(len(turtle_mask)):
            mask1 += self.coco.annToMask(turtle_mask[i])
        mask2 = np.zeros_like(mask1)
        for i in range(len(leg_mask)):
            mask2 += self.coco.annToMask(leg_mask[i])
        mask3 = np.zeros_like(mask2)
        for i in range(len(head_mask)):
            mask3 += self.coco.annToMask(head_mask[i])
        mask3[mask3 > 0] = 2
        mask2[mask2 > 0] = 1
        mask1[mask1 > 0] = 1
        mask = mask1 + mask2 
        mask[mask > 1] = 2
        mask = mask + mask3
        mask[mask > 2] = 3
        return mask


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self.coco.loadImgs([image_id])[0]
        image = cv.imread(f"{self.dataset_path}/{image['file_name']}")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = self._process_mask(image_id, image)
        image = cv.resize(image, self.resize)
        mask = cv.resize(mask, self.resize)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask

def seprate_train_val_test(coco: COCO, test_size: float = 0.1, val_size: float = 0.2, random_state: int = 42):
    train_ids, test_ids = train_test_split(coco.getImgIds()[:2000], test_size=test_size, random_state=random_state)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size, random_state=random_state)
    return train_ids, val_ids, test_ids
