from pycocotools.coco import COCO
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def visualize(figsize=(16,5), **images):
    n = len(images)
    plt.figure(figsize=figsize)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def visualize_image(id: int, coco: COCO, mask=False, cat_filter=None, dataset_path="data"):
    image = coco.loadImgs([id])[0]
    image = cv.imread(f"{dataset_path}/{image['file_name']}")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cat_ids = coco.getCatIds()
    if cat_filter:
        cat_ids = coco.getCatIds(catNms=cat_filter)
    ann_ids = coco.getAnnIds(imgIds=id, catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    if mask:
        plt.imshow(image)
        coco.showAnns(anns)
    else:
        for ann in anns:
            x, y, w, h = ann['bbox']
            cv.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
        visualize(image=image)

def visualize_image_with_mask(image, mask, pred=None):
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    if pred is None:
        visualize(figsize=(10, 5), original_image=image, mask=mask)
    else:
        visualize(figsize=(10, 5), original_image=image, true_mask=mask, pred_mask=pred)
