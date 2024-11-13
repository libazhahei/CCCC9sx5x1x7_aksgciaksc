import argparse
import os
import torch
from networks.loaders import ModelSelector, import_necessary
from utilis.dataset import TurtlesDataset
from pycocotools.coco import COCO
from utilis.preprocessing import get_valid_preprocessing
import numpy as np
from utilis.visualization import store_image_with_mask
def main(args):
    import_necessary(args.model_name)
    coco = COCO(f"{args.data_dir}/annotations.json")
    test_dataset = TurtlesDataset(coco, coco.getImgIds(), resize=(args.size, args.size), dataset_path=args.data_dir, transform=get_valid_preprocessing())
    selector = ModelSelector(args.num_classes, pretrained=True)
    model, get_predition = selector.select_model(args.model_name, checkpoint_path=args.checkpoint)
    if model is None:
        raise ValueError(f"Model {args.model_name} not found, or not implemented")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.img_id is not None:
        image, mask = test_dataset[args.img_id]
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        image = image.to(device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            pred_mask = get_predition(model.forward(image))
            pred_mask = torch.argmax(pred_mask, dim=1)
            pred_mask = pred_mask.cpu().numpy()
            pred_mask = np.squeeze(pred_mask)
            mask = mask.cpu().numpy()
            mask = np.squeeze(mask)
            store_image_with_mask(image[0], mask, pred_mask, args.img_dir, args.model_name, args.img_id)
        return
    np.random.seed(args.seed)
    for _ in range(args.n_img):
        random_id=np.random.randint(0, len(test_dataset))
        image, mask = test_dataset[random_id]
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        image = image.to(device)
        model.eval()
        with torch.no_grad():
            pred_mask = get_predition(model.forward(image))
            pred_mask = torch.argmax(pred_mask, dim=1)
            pred_mask = pred_mask.cpu().numpy()
            pred_mask = np.squeeze(pred_mask)
            mask = mask.cpu().numpy()
            mask = np.squeeze(mask)
            print("Image id:", random_id)
            store_image_with_mask(image[0], mask, pred_mask, args.img_dir, args.model_name, random_id)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, default="/content/dataset/turtles-data/data")
    arg.add_argument("--model_dir", type=str, default="models")
    arg.add_argument("--img_dir", type=str, required="img")
    arg.add_argument("--model_name", type=str, required=True)
    arg.add_argument("--size", type=int, default=1024)
    arg.add_argument("--model_version", type=int, default=-1)
    arg.add_argument("--batch_size", type=int, default=4)
    arg.add_argument("--num_classes", type=int, default=4)
    arg.add_argument("--checkpoint", type=str, default=None)
    arg.add_argument("--n_img", type=int, default=20)
    arg.add_argument("--img_id", type=int, default=None)
    arg.add_argument("--seed", type=int, default=42)
    args = arg.parse_args()
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)
    if not os.path.exists(args.model_dir) and args.checkpoint is None:
        raise ValueError(f"Model directory {args.model_dir} does not exist")
    if args.checkpoint is None and args.model_version == -1:
        args.model_version = len(os.listdir(args.model_dir))
    if args.checkpoint is None and not os.path.exists(f"{args.model_dir}/{args.model_version}"):
        raise ValueError(f"Model version {args.model_version} does not exist")
    if args.checkpoint is None:
        args.checkpoint = f"{args.model_dir}/{args.model_version}/{args.model_name}_best_model.pth"
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint {args.checkpoint} does not exist")
    main(args)