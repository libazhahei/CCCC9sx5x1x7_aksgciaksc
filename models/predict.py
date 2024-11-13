import argparse
import os
import torch
from networks.loaders import ModelSelector, import_necessary
from utilis.epochs import  ValidEpoch
from utilis.metrics import MultiClassIoU, MultiClassF1Score, MultiClassPrecision, MultiClassRecall
from utilis.dataset import TurtlesDataset, seprate_train_val_test
from pycocotools.coco import COCO
from torch.utils.data import DataLoader 
from utilis.preprocessing import get_test_preprocessing
from utilis.loss import MultiClassCombinedLoss
import pandas as pd

def main(args):
    import_necessary(args.model_name)
    coco = COCO(f"{args.data_dir}/annotations.json")
    _, _, test_ids = seprate_train_val_test(coco, test_size=0.2, val_size=0.2, random_state=42)
    test_dataset = TurtlesDataset(coco, test_ids, resize=(args.size, args.size), dataset_path=args.data_dir, transform=get_test_preprocessing())
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
    loss = MultiClassCombinedLoss(num_classes=args.num_classes, alpha=args.alpha, beta=args.beta, gamma=args.gamma, class_weights=torch.tensor([0.02, 0.21, 0.63, 1.0]))
    test_epoch = ValidEpoch(
        model=model,
        loss=loss,
        metrics=test_metrics,
        device=device,
        verbose=True,
        get_prediction=get_predition,
    )
    print("Start Testing")
    test_logs = test_epoch.run(test_loader)
    for key, value in test_logs.items():
        print(f"{key}: {value}")
    test_logs_df = pd.DataFrame(test_logs, index=[0])
    if args.model_version == -1:
        test_logs_df.to_csv(f"{args.model_dir}/{args.model_name}_test_metrics.csv", index=False)
    else:
        test_logs_df.to_csv(f"{args.model_dir}/{args.model_name}_{args.model_version}/{args.model_name}_test_metrics.csv", index=False)

    pass 
if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, default="/content/dataset/turtles-data/data")
    arg.add_argument("--model_dir", type=str, default="models")
    arg.add_argument("--model_name", type=str, required=True)
    arg.add_argument("--size", type=int, default=1024)
    arg.add_argument("--model_version", type=int, default=-1)
    arg.add_argument("--batch_size", type=int, default=4)
    arg.add_argument("--num_classes", type=int, default=4)
    arg.add_argument("--checkpoint", type=str, default=None)
    arg.add_argument("--alpha", type=float, default=0.2)
    arg.add_argument("--beta", type=float, default=0.3)
    arg.add_argument("--gamma", type=float, default=0.5)
    arg.add_argument("--focal_gamma", type=float, default=2.0)
    args = arg.parse_args()
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    if not os.path.exists(args.model_dir) and args.checkpoint is None:
        raise ValueError(f"Model directory {args.model_dir} does not exist")
    if args.checkpoint is None and args.model_version == -1:
        length = len(os.listdir(args.model_dir))
        args.model_version = len(os.listdir(args.model_dir))
        if os.path.exists(f"{args.model_dir}/{args.model_name}_{args.model_version}"):
            args.checkpoint = f"{args.model_dir}/{args.model_name}_{args.model_version}/{args.model_name}_best_model.pth"
    if args.checkpoint is None:
        raise ValueError(f"Checkpoint for model does not exist")
    if not os.path.exists(f"{args.model_dir}"):
        os.makedirs(f"{args.model_dir}")
    if args.model_version != -1 and not os.path.exists(f"{args.model_dir}/{args.model_name}_{args.model_version}"):
        raise ValueError(f"Model version {args.model_version} does not exist")
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint {args.checkpoint} does not exist")
    main(args)