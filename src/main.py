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
    parser.add_argument('moving', type=ap.dir_path, help='Path to directory of moving images')
    parser.add_argument('fixed', type=ap.file_path, help='Path to fixed image')
    parser.add_argument('moving_labels', type=ap.dir_path, help='Path to directory of moving image labels')
    parser.add_argument('fixed_label', type=ap.file_path, help='Path to fixed label')
    parser.add_argument('--weights_file', type=ap.file_path, help='Load model weights from file')
    parser.add_argument('--save_weights_file', default=ap.join(ap.MODEL_PATH, 'model.pth'), \
        help='Save model weights to file')
    parser.add_argument('--history', type=ap.save_image_path, default=ap.join(ap.VISUALIZE_PATH, 'history.png'), \
        help='Path to save model history image')
    parser.add_argument('--history_log', type=ap.save_file_path, default=ap.join(ap.VISUALIZE_PATH, 'history.pkl'), \
        help='Path to save model history log object')
    
    # Training & data loading arguments
    parser.add_argument("--resize_ratio", type=float, help="Ratio to which the data is resized, e.g., 0.5 with shape (100, 150, 50) -> (50, 75, 25)")
    parser.add_argument("--resize_shape", type=ap.delimited_ints, help="Shape to which the data is resized (a string of comma-delimited integers). May not be exactly this shape, but very similar")
    parser.add_argument("--cache_rate", type=ap.percent, default=1, help="Percentage of training data to load/cache at once - takes min of cache_rate and cache_num")
    parser.add_argument("--cache_num", type=int, default=sys.maxsize, help="Number of training samples to load/cache at once - takes min of cache_rate and cache_num")
    parser.add_argument("--verbose", action="store_true", help="If flagged, the program will convey more information through stdout")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size - highly recommended to use the default of 1")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--val_interval", type=int, default=1, help="Calculatge validation every x epochs during training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lr_factor", type=ap.percent, default=0.5, help="Learning rate percentage decrease factor")
    parser.add_argument("--lr_patience", type=int, default=5, help="Number of epochs between checking to decrease learning rate")
    parser.add_argument("--es_patience", type=int, default=10, help="Number of epochs between early stopping checks")
    parser.add_argument("--val_percent", type=ap.percent, default=0.15, help="Validation dataset percentage")
    parser.add_argument("--test_percent", type=ap.percent, default=0.15, help="Test dataset percentage")
    parser.add_argument("--test", action="store_true", help="If flagged, test at the end of training")
    # Note, if os.cpu_count() - 1 = 0, it will just use the main process
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() - 1, \
        help="Number of workers to perform multi-threading during caching. Defaults to the number of CPUs - 1. Value 0 uses no multi-threading")
    parser.add_argument("--deterministic", action="store_true", help="If flagged, have deterministic training")

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Print MONAI library config    
    print_config()

    # Format data
    data_dicts = dl.format_data(args.moving, args.fixed, \
        moving_labels=args.moving_labels, fixed_label=args.fixed_label)
    
    # Split into training, validation, and testing
    train_files, val_files, test_files = dl.split_dataset(data_dicts, \
        val_percent=args.val_percent, test_percent=args.test_percent)

    print("Number of training files:", len(train_files))
    print("Number of validation files:", len(val_files))
    print("Number of testing files:", len(test_files))

    # If deterministic flagged, set seed to make training deterministic
    if args.deterministic:
        set_determinism(seed=0)

    # Create dataloaders
    train_loader, val_loader = dl.create_train_dataloaders(args, train_files, \
        val_files, visualize=True, \
        visualize_path=ap.join(ap.VISUALIZE_PATH, "data.png"))
    
    # Get device
    device = m.get_device()

    # Define model
    model = m.Model(device, val_interval=args.val_interval, lr=args.lr, \
        lr_factor=args.lr_factor, lr_patience=args.lr_patience, \
        es_patience=args.es_patience)

    # If args.weights is specified, load the weights from file
    if args.weights_file is not None:
        model.load_weights(args.weights_file)

    # Train model
    history = model.train(train_loader, val_loader, device, \
        args.max_epochs, save_weights_file=args.save_weights_file)

    # Save history
    with open(args.history_log, 'wb') as f:
        pickle.dump(history, f)

    # Code to load the pickled history object
    #with open(args.history_log, 'rb') as f:
        #history = pickle.load(f)

    # Plot history
    vis.plot_history(history, save_path=args.history, val_interval=args.val_interval)

    # Perform inference on testing data
    if args.test:
        if args.test_percent == 0:
            print("Testing data split percentage (test_percent) cannot be 0 when testing")
        else:
            test_loader = create_dataloader_infer(test_files, args.fixed, \
                resize_shape=args.resize_shape, resize_ratio=args.resize_ratio)
            model.infer(test_loader, device, \
                visualize_save_path=ap.join(ap.VISUALIZE_PATH, "infer.png"))


if __name__ == '__main__':
    main()