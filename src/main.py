import os
import torch
from monai.config import print_config
from monai.utils import set_determinism

# Local imports
import dataloader as dl
import model as m
import argparsing as ap
import visualize as vis


def get_device(verbose=True):
        """
        Get the device on which to train.
        Use a GPU if possible, otherwise CPU.
        """
        print("torch.cuda.is_available()", torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if verbose:
            if device.type == 'cuda':
                print("Using Device:", torch.cuda.get_device_name(0))
            else:
                print("Using Device:", device)
        
        return device


def get_files(args):
    # Get all non-label files
    images = os.listdir(args.images)
    labels = os.listdir(args.labels)

    images = sorted(images)
    labels = sorted(labels)

    # Assert the images and labels folders have the same files
    assert len(images) == len(labels) and images == labels

    data_dicts = []
    for i, image in enumerate(images):
        data_dicts.append(
            {
                "fixed_image": args.atlas_file,
                "moving_image": os.path.join(args.images, image),
                "fixed_label": args.atlas_label_file,
                "moving_label": os.path.join(args.images, labels[i]),
            }
        )
    
    return data_dicts


def main():
    # Parse arguments
    args = ap.parse_arguments_main()
    
    # Print MONAI library config    
    print_config()

    # Get data files
    data_dicts = get_files(args)
    
    # Split into training, validation, and testing
    train_files, val_files, test_files = dl.split_dataset(data_dicts, \
        val_percent=args.val_percent, test_percent=args.test_percent)

    print("len(train_files)", len(train_files))
    print("len(val_files)", len(val_files))

    # If deterministic flagged, set seed to make training deterministic
    if args.deterministic:
        set_determinism(seed=0)

    # Create dataloaders
    train_loader, val_loader = dl.create_dataloaders(args, train_files, val_files, \
        visualize=True, visualize_path=ap.join(ap.VISUALIZE_PATH, "data.png"))
    
    # Get device
    device = get_device()#torch.device("cuda:0")

    # Define model
    model = m.Model(args, device)

    # If args.weights is specified, load the weights from file
    if args.weights_file is not None:
        model.load_weights(args.weights_file)

    # Train model
    epoch_loss_values, metric_values = model.train(args, train_loader, val_loader, device, ap.MODEL_PATH)

    # Plot history
    vis.plot_history(args, epoch_loss_values, metric_values, ap.join(ap.VISUALIZE_PATH, "history.png"))

    # Perform inference
    model.infer_val(val_loader, device, visualize_save_path=ap.join(ap.VISUALIZE_PATH, "infer.png"))


if __name__ == '__main__':
    main()