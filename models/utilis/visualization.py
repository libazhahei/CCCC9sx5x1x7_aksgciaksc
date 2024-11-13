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

def store_image_with_mask(image, mask, pred, path, model_name, img_id):
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    image = cv.resize(image, (2000, 1333))
    mask = cv.resize(mask, (2000, 1333))
    pred = cv.resize(pred, (2000, 1333))
    plt.imsave(f"{path}/{model_name}_image_{img_id}.jpg", image)
    plt.imsave(f"{path}/{model_name}_mask_{img_id}.jpg", mask)
    plt.imsave(f"{path}/{model_name}_pred_{img_id}.jpg", pred)

def mega_draw(images, masks, preds, model_names):
    # Draw a graph of 3 rows, 4 cols, each col is a model, rows are img, mask, and pred prepsective;y
    # Hiden the title of each image, put the model name at the top of each col
    # On the left hand side, put the image, in the middle put the mask, on the right put the prediction
    # the border and the padding between the images should be small
    fig, axs = plt.subplots(4, 4, figsize=(20, 12))
    for i, (image, mask, pred, model_name) in enumerate(zip(images, masks, preds, model_names)):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        diff = mask_diff(mask, pred)
        axs[0, i].imshow(image)
        axs[1, i].imshow(mask)
        axs[2, i].imshow(pred)
        axs[0, i].set_title(model_name)
        axs[3, i].imshow(diff)
    for ax in axs.flat:
        ax.label_outer()
        ax.axis("off")
    
def mask_diff(masks, pred):
    # compare the difference between the mask and the prediction
    # Any difference between the mask and the prediction should be colored with white
    # The indifference should be colored with black
    diff = masks - pred
    diff[diff != 0] = 255
    return diff

    