# User Manual

This manual provides detailed instructions on how to train, predict, and visualize using the provided scripts. The scripts are designed for a turtle segmentation task using deep learning models.

## Table of Contents
1. [Training](#training)
2. [Prediction](#prediction)
3. [Visualization](#visualization)
4. [Parameters](#parameters)
5. [Supported Models](#supported-models)
6. [File Structure](#file-structure)

## Training

To train a model, use the `train.py` script. This script will train a model on the provided dataset and save the best model based on validation IoU score.

### Command
```bash
python train.py --data_dir <data_directory> --model_name <model_name> [options]
```

### Example
```bash
python train.py --data_dir /content/dataset/turtles-data/data --model_name unet --epochs 50
```

## Prediction

To predict using a trained model, use the `predict.py` script. This script will evaluate the model on the test set and save the metrics.

### Command
```bash
python predict.py --data_dir <data_directory> --model_name <model_name> --checkpoint <checkpoint_path> [options]
```

### Example
```bash
python predict.py --data_dir /content/dataset/turtles-data/data --model_name unet --checkpoint models/unet_best_model.pth
```

## Visualization

To visualize the predictions, use the `visualization.py` script. This script will generate and save images with predicted masks.

### Command
```bash
python visualization.py --data_dir <data_directory> --model_name <model_name> --checkpoint <checkpoint_path> --img_dir <image_directory> [options]
```

### Example
```bash
python visualization.py --data_dir /content/dataset/turtles-data/data --model_name unet --checkpoint models/unet_best_model.pth --img_dir output_images
```

## Parameters

### Common Parameters
- `--data_dir`: Directory containing the dataset.
- `--model_name`: Name of the model to use (e.g., `unet`).
- `--checkpoint`: Path to the model checkpoint file.

### Training Parameters
- `--model_dir`: Directory to save the models (default: `models`).
- `--size`: Size to resize the images (default: `1024`).
- `--model_version`: Version of the model (default: `-1`).
- `--batch_size`: Batch size for training (default: `8`).
- `--num_classes`: Number of classes (default: `4`).
- `--epochs`: Number of epochs to train (default: `10`).
- `--early_stop`: Early stopping patience (default: `5`).
- `--save_step`: Save model every n epochs (default: `10`).
- `--base_lr`: Base learning rate (default: `0.0001`).
- `--max_lr`: Maximum learning rate (default: `0.001`).
- `--step_size`: Step size for cyclic learning rate (default: `1000`).
- `--cycle_momentum`: Use cycle momentum (default: `False`).
- `--amp`: Use automatic mixed precision (default: `False`).
- `--resume`: Resume training from checkpoint (default: `False`).
- `--alpha`: Alpha parameter for loss function (default: `0.2`).
- `--beta`: Beta parameter for loss function (default: `0.3`).
- `--gamma`: Gamma parameter for loss function (default: `0.5`).
- `--focal_gamma`: Gamma parameter for focal loss (default: `2.0`).

### Prediction Parameters
- `--batch_size`: Batch size for prediction (default: `4`).

### Visualization Parameters
- `--img_dir`: Directory to save the images.
- `--n_img`: Number of images to visualize (default: `20`).
- `--img_id`: Specific image ID to visualize (default: `None`).
- `--seed`: Random seed (default: `42`).

By following these instructions, you can effectively train, predict, and visualize using the provided scripts. Make sure to adjust the parameters as needed for your specific use case.
## Supported Models

The following models are supported for the turtle segmentation task:

- `deeplabv3_baseline`
- `deeplabv3_sp`
- `deeplabv3_mrb`
- `deeplabv3_all`
- `maskrcnn`
- `unet`
- `res_unet++`
- `attention_unet`

Each model can be selected using the `--model_name` parameter in the scripts.

## File Structure

The following is an example of the file structure for the project:

```
project_root/
│
├── data/
│   ├── annotations.json
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── model_version_1/
│   │   ├── unet_best_model.pth
│   │   └── unet_epoch_10.pth
│   └── model_version_2/
│       ├── unet_best_model.pth
│       └── unet_epoch_20.pth
│
├── output_images/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
│
├── train.py
├── predict.py
├── visualization.py
├── networks/
│   ├── loaders.py
│   └── ...
├── utilis/
│   ├── epochs.py
│   ├── loss.py
│   ├── metrics.py
│   ├── dataset.py
│   └── preprocessing.py
└── README.md
```

- `data/`: Contains the dataset and annotations.
- `models/`: Contains the saved models organized by version.
- `output_images/`: Contains the visualized output images.
- `train.py`: Script for training the model.
- `predict.py`: Script for predicting using the trained model.
- `visualization.py`: Script for visualizing the predictions.
- `networks/`: Contains the model loaders and other network-related scripts.
- `utilis/`: Contains utility scripts for epochs, loss functions, metrics, dataset handling, and preprocessing.
- `README.md`: The user manual and documentation.