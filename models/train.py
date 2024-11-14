import argparse
import os
import torch
from networks.loaders import ModelSelector
from utilis.epochs import TrainEpoch, ValidEpoch
from utilis.loss import MultiClassCombinedLoss
from utilis.metrics import MultiClassIoU
from utilis.dataset import TurtlesDataset, seprate_train_val_test
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from utilis.preprocessing import get_valid_preprocessing
import segmentation_models_pytorch.utils as smp_utils

import pandas as pd
def main(args):
    coco = COCO(f"{args.data_dir}/annotations.json")
    train_ids, val_ids, _ = seprate_train_val_test(coco, test_size=0.2, val_size=0.2, random_state=42)
    train_dataset = TurtlesDataset(coco, train_ids, resize=(args.size, args.size), transform=get_valid_preprocessing())
    val_dataset = TurtlesDataset(coco, val_ids, resize=(args.size, args.size), transform=get_valid_preprocessing())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    selector = ModelSelector(args.num_classes, pretrained=args.resume)
    model, get_predition = selector.select_model(args.model_name, checkpoint_path=args.checkpoint)
    if model is None:
        raise ValueError(f"Model {args.model_name} not found, or not implemented")
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])
    loss = MultiClassCombinedLoss(num_classes=args.num_classes, alpha=args.alpha, beta=args.beta, gamma=args.gamma, class_weights=torch.tensor([0.02, 0.21, 0.63, 1.0]))
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=1000, cycle_momentum=False)
    metrics = [
        MultiClassIoU(args.num_classes),
        MultiClassIoU(args.num_classes, single_calss_id=(0, 'background')),
        MultiClassIoU(args.num_classes, single_calss_id=(1, 'turtle')),
        MultiClassIoU(args.num_classes, single_calss_id=(2, 'legs')),
        MultiClassIoU(args.num_classes, single_calss_id=(3, 'head')),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_epoch = TrainEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        device=device,
        verbose=True,
        amp=False,
        get_prediction=get_predition,
    )
    val_epoch = ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
        get_prediction=get_predition,
    )
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []
    not_best=0

    for i in range(0, args.epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = val_epoch.run(val_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        if best_iou_score < valid_logs['MultiClassIoU']:
            best_iou_score = valid_logs['MultiClassIoU']
            torch.save(model.state_dict(), f'./{args.model_dir}/{args.model_name}_{args.model_version}/{args.model_name}_best_model.pth')
            print('Model saved!')
            not_best = 0
        else:
            if not_best > args.early_stop:
                break
            not_best += 1
        if i % args.save_step == 0:
            torch.save(model.state_dict(), f'./{args.model_dir}/{args.model_name}_{args.model_version}/{args.model_name}_epoch_{i}.pth')
            
    pd.DataFrame(train_logs_list).to_csv(f'./{args.model_dir}/{args.model_name}_{args.model_version}/{args.model_name}_train_logs.csv', index=False)
    pd.DataFrame(valid_logs_list).to_csv(f'./{args.model_dir}/{args.model_name}_{args.model_version}/{args.model_name}_valid_logs.csv', index=False)
    torch.save(model.state_dict(), f'./{args.model_dir}/{args.model_name}_{args.model_version}/{args.model_name}_last_model.pth')

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, default="data")
    arg.add_argument("--model_dir", type=str, default="models")
    arg.add_argument("--model_name", type=str, required=True)
    arg.add_argument("--size", type=int, default=1024)
    arg.add_argument("--model_version", type=int, default=-1)
    arg.add_argument("--batch_size", type=int, default=8)
    arg.add_argument("--num_classes", type=int, default=4)
    arg.add_argument("--epochs", type=int, default=10)
    arg.add_argument("--early_stop", type=int, default=5)
    arg.add_argument("--save_step", type=int, default=10)
    arg.add_argument("--base_lr", type=float, default=0.0001)
    arg.add_argument("--max_lr", type=float, default=0.001)
    arg.add_argument("--step_size", type=int, default=1000)
    arg.add_argument("--cycle_momentum", type=bool, default=False)
    arg.add_argument("--amp", type=bool, default=False)
    arg.add_argument("--resume", type=bool, default=False)
    arg.add_argument("--checkpoint", type=str, default=None)
    arg.add_argument("--alpha", type=float, default=0.2)
    arg.add_argument("--beta", type=float, default=0.3)
    arg.add_argument("--gamma", type=float, default=0.5)
    arg.add_argument("--focal_gamma", type=float, default=2.0)
    args = arg.parse_args()
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if args.model_version == -1:
        args.model_version = len(os.listdir(args.model_dir)) + 1
    if args.resume and args.checkpoint is None:
        raise ValueError("Checkpoint path is required for resuming training")
    if not os.path.exists(f"{args.model_dir}/{args.model_version}"):
        os.makedirs(f"{args.model_dir}/{args.model_version}")
    main(args)

