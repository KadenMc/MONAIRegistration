import torch

# Local imports
import argparse
import argparsing as ap
import model as m
from dataloader import create_dataloader_infer

def parse_arguments():
    """
    Parse arguments for infererence.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('moving_file', type=ap.file_path, help='Path to moving image')
    parser.add_argument('moving_labels_file', type=ap.file_path, help='Path to moving image')
    parser.add_argument('fixed_file', type=ap.file_path, help='Path to atlas file')
    parser.add_argument('fixed_labels_file', type=ap.file_path, help='Path to fixed labels file')
    parser.add_argument('weights_file', type=ap.file_path, help='Load model weights from file')
    
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Define data
    data_dict = {
            "fixed_image": args.fixed_file,
            "moving_image": args.moving_file,
            "fixed_label": args.fixed_labels_file,
            "moving_label": args.moving_labels_file,
    }
    
    # Define data loader
    loader = create_dataloader_infer(data_dict)

    # Define device
    device = m.get_device()

    # Define model
    args.lr = 0
    model = m.Model(args, device)
    model.load_weights(args.weights_file)

    # Infer
    model.infer_val(loader, device)


if __name__ == "__main__":
    main()