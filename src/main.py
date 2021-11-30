import os
import sys
from monai.config import print_config
from monai.utils import set_determinism
import argparse

# Local imports
import dataloader as dl
import model as m
import argparsing as ap
import visualize as vis

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('images', type=ap.dir_path, help='Path to folder of images')
    parser.add_argument('labels', type=ap.dir_path, help='Path to folder of labels')
    parser.add_argument('atlas', type=ap.file_path, help='Path to atlas file')
    parser.add_argument('atlas_label', type=ap.file_path, help='Path to atlas file')
    parser.add_argument('--weights_file', type=ap.file_path, help='Load model weights from file')
    parser.add_argument('--save_weights_file', default=ap.join(ap.MODEL_PATH, 'model.pth'), \
        help='Save model weights to file')
    parser.add_argument('--history', default=ap.join(ap.VISUALIZE_PATH, 'history.png'), \
        help='Path to save model history')
    
    # Training & data loading arguments
    parser.add_argument("--resample_ratio", type=float, help="Ratio to which the data is resampled, e.g., 0.5 with shape (100, 150, 50) -> (50, 75, 25)")
    parser.add_argument("--resample_shape", type=ap.delimited_ints, help="Shape to which the data is resampled")
    parser.add_argument("--cache_rate", type=ap.percent, default=1, help="Percentage of training data to load/cache at once - takes min of cache_rate and cache_num")
    parser.add_argument("--cache_num", type=int, default=sys.maxsize, help="Number of training samples to load/cache at once - takes min of cache_rate and cache_num")
    parser.add_argument("--verbose", action="store_true", help="If flagged, the program will convey more information through stdout")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size - highly recommended to use the default of 1")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--val_interval", type=int, default=1, help="Calculatge validation every x epochs during training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--val_percent", type=ap.percent, default=0.15, help="Validation dataset percentage")
    parser.add_argument("--test_percent", type=ap.percent, default=0.15, help="Test dataset percentage")
    parser.add_argument("--num_workers", type=int, default=0, \
        help="Number of workers to perform multi-threading during caching. Default of 0 uses no multi-threading")
    parser.add_argument("--deterministic", action="store_true", help="If flagged, have deterministic training")

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Print MONAI library config    
    print_config()

    # Format data
    data_dicts = dl.format_data(args.images, args.labels, args.atlas, args.atlas_label)
    
    # Split into training, validation, and testing
    train_files, val_files, test_files = dl.split_dataset(data_dicts, \
        val_percent=args.val_percent, test_percent=args.test_percent)

    print("Number of training files:", len(train_files))
    print("Number of validation files:", len(val_files))

    # If deterministic flagged, set seed to make training deterministic
    if args.deterministic:
        set_determinism(seed=0)

    # Create dataloaders
    train_loader, val_loader = dl.create_dataloaders(args, train_files, val_files, \
        visualize=True, visualize_path=ap.join(ap.VISUALIZE_PATH, "data.png"))
    
    # Get device
    device = m.get_device()

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
    #model.infer_val(val_loader, device, visualize_save_path=ap.join(ap.VISUALIZE_PATH, "infer.png"))


if __name__ == '__main__':
    main()