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


# Customised dataset
class TurtleDataset(Dataset):
    def __init__(self, img_dir, coco, transform=None):
        self.img_dir = img_dir
        self.coco = coco
        self.transform = transform
        self.img_ids = list(self.coco.getImgIds())
        self.cat_ids = self.coco.getCatIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        img = cv.imread(self.img_dir + self.coco.loadImgs([img_id])[0]['file_name'])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img = img.resize((1024, 1024))

        boxes = []
        labels = []
        masks = []

        # Mask RCNN needs boxes, labels and masks
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            x, y, width, height = ann['bbox']
            boxes.append([x, y, x + width, y + height])
            labels.append(ann['category_id'])
            masks.append(self.coco.annToMask(ann))
        from sklearn.preprocessing import MinMaxScaler

        boxes = MinMaxScaler((0, 1024)).fit_transform(np.array(boxes))
        masks = cv.resize(masks, (1024, 1024))    

        # Transform into boxes, labels and masks
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(masks, dtype=torch.uint8),
            # 'id': torch.tensor([img_id]),
        }


        return img, target

# Process the list of sample from batch
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

dataset = TurtleDataset(img_dir=img_dir, coco=coco, transform=transform)

# sample image
sample_image(dataset, 321)

# Split dataset - Ensure Open-set splitting
split = pd.read_csv(split_file)
train_data = Subset(dataset, list(split[split['split_open'] == 'train'].index))
test_data = Subset(dataset, list(split[split['split_open'] == 'test'].index))
valid_data = Subset(dataset, list(split[split['split_open'] == 'valid'].index))

# Reduced dataset
# proportion = 0.01
# test_prop = 0.01
proportion = 0.7
test_prop = 0.7
sub_ind = random.sample(range(len(train_data)), int(proportion * len(train_data)))
train_dataset = DataLoader(Subset(train_data, sub_ind), batch_size=batch, shuffle=True, collate_fn=collate_fn)

sub_ind = random.sample(range(len(valid_data)), int(test_prop * len(valid_data)))
valid_dataset = DataLoader(Subset(valid_data, sub_ind), batch_size=1, collate_fn=collate_fn)

sub_ind = random.sample(range(len(test_data)), int(test_prop * len(test_data)))
test_dataset = DataLoader(Subset(test_data, sub_ind), batch_size=1, collate_fn=collate_fn)

print("Full training set size: ", len(train_data))
print("Full validation set size: ", len(valid_data))
print("Full test set size: ", len(test_data))
print("Reduced training dataset size:", len(train_dataset)*batch)
print("Reduced validation dataset size:", len(valid_dataset))
print("Reduced test dataset size:", len(test_dataset))