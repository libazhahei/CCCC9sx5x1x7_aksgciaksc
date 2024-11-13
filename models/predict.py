import argparse
import os
import torch
from networks.loaders import ModelSelector
from utilis.epochs import  ValidEpoch
from utilis.metrics import MultiClassIoU, MultiClassF1Score, MultiClassPrecision, MultiClassRecall
from utilis.dataset import TurtlesDataset, seprate_train_val_test
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import pandas as pd
def main(args):
    coco = COCO(f"{args.data_dir}/annotations.json")
    _, _, test_ids = seprate_train_val_test(coco, test_size=0.2, val_size=0.2, random_state=42)
    test_dataset = TurtlesDataset(coco, test_ids, resize=(args.size, args.size))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    selector = ModelSelector(args.num_classes, pretrained=True)
    model, get_predition = selector.select_model(args.model_name, checkpoint_path=args.checkpoint)
    if model is None:
        raise ValueError(f"Model {args.model_name} not found, or not implemented")
    test_metrics = [
        MultiClassIoU(4),
        MultiClassIoU(4, single_calss_id=(0, 'background')),
        MultiClassIoU(4, single_calss_id=(1, 'turtle')),
        MultiClassIoU(4, single_calss_id=(2, 'legs')),
        MultiClassIoU(4, single_calss_id=(3, 'head')),
        MultiClassF1Score(4),
        MultiClassF1Score(4, single_calss_id=(0, 'background')),
        MultiClassF1Score(4, single_calss_id=(1, 'turtle')),
        MultiClassF1Score(4, single_calss_id=(2, 'legs')),
        MultiClassF1Score(4, single_calss_id=(3, 'head')),
        MultiClassPrecision(4),
        MultiClassPrecision(4, single_calss_id=(0, 'background')),
        MultiClassPrecision(4, single_calss_id=(1, 'turtle')),
        MultiClassPrecision(4, single_calss_id=(2, 'legs')),
        MultiClassPrecision(4, single_calss_id=(3, 'head')),
        MultiClassRecall(4),
        MultiClassRecall(4, single_calss_id=(0, 'background')),
        MultiClassRecall(4, single_calss_id=(1, 'turtle')),
        MultiClassRecall(4, single_calss_id=(2, 'legs')),
        MultiClassRecall(4, single_calss_id=(3, 'head')),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_epoch = ValidEpoch(
        model=model,
        loss=None,
        metrics=test_metrics,
        device=device,
        verbose=True,
        get_prediction=get_predition,
    )
    test_logs = test_epoch.run(test_loader)
    for key, value in test_logs.items():
        print(f"{key}: {value}")
    pd.DataFrame(test_logs).to_csv(f"{args.model_dir}/{args.model_version}/{args.model_name}_test_metrics.csv", index=False)

    pass 
if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, default="data")
    arg.add_argument("--model_dir", type=str, default="models")
    arg.add_argument("--model_name", type=str, required=True)
    arg.add_argument("--size", type=int, default=1024)
    arg.add_argument("--model_version", type=int, default=-1)
    arg.add_argument("--batch_size", type=int, default=8)
    arg.add_argument("--num_classes", type=int, default=4)
    arg.add_argument("--checkpoint", type=str, default=None)
    args = arg.parse_args()
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory {args.model_dir} does not exist")
    if args.model_version == -1:
        args.model_version = len(os.listdir(args.model_dir))
    if not os.path.exists(f"{args.model_dir}/{args.model_version}"):
        raise ValueError(f"Model version {args.model_version} does not exist")
    if args.checkpoint is None:
        args.checkpoint = f"{args.model_dir}/{args.model_version}/{args.model_name}_best_model.pth"
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint {args.checkpoint} does not exist")
    main(args)