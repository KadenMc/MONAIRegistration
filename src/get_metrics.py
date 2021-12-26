import argparse
import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MSEMetric

# Local imports
import argparsing as ap
from model import get_device
from dataloader import format_data, create_dataloader_infer


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data', type=ap.path, help='Path to image or directory of images')
    parser.add_argument('fixed', type=ap.file_path, help='Path to fixed image')
    parser.add_argument('--labels', type=ap.path, help='Path to image label or directory of image labels')
    parser.add_argument('--fixed_label', type=ap.file_path, help='Path to fixed label')

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arguments()

    # Get device
    device = get_device()

    # Format the data
    data_dicts = format_data(args.data, args.fixed, \
        moving_labels=args.labels, fixed_label=args.fixed_label)

    # Get dataloader
    loader = create_dataloader_infer(data_dicts, args.fixed)

    # Define metrics
    hausdorff_metric = HausdorffDistanceMetric()
    mse_metric = MSEMetric()
    dice_metric = DiceMetric(include_background=True, reduction="mean", \
        get_not_nans=False)

    # Get metrics
    for data in loader:
        # Calculate image metrics
        image = data["moving_image"].to(device)
        fixed_image = data["fixed_image"].to(device)
        mse_metric(y_pred=image, y=fixed_image)
        
        # Optionally calculate label metrics
        if "moving_label" in data and "fixed_label" in data:
            label = data["moving_label"].to(device).byte()
            fixed_label = data["fixed_label"].to(device).byte()
            dice_metric(y_pred=label, y=fixed_label)
            hausdorff_metric(y_pred=label, y=fixed_label)

    # Get average of each metric
    metrics = dict()
    metrics["MSE"] = mse_metric.aggregate().item()
    
    try:
        metrics["DICE"] = dice_metric.aggregate().item()
        metrics["Hausdorff"] = hausdorff_metric.aggregate().item()
    except TypeError:
        pass

    for m in metrics:
        if len(data_dicts) == 1:
            print("{}: {}".format(m, metrics[m]))
        else:
            print("Mean {}: {}".format(m, metrics[m]))


if __name__ == '__main__':
    main()