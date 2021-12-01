# Local imports
import argparse
import argparsing as ap
import model as m
import dataloader as dl

def parse_arguments():
    """
    Parse arguments for infererence.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('moving', type=ap.file_or_dir_path, help='Path to moving image file or directory')
    parser.add_argument('moving_labels', type=ap.file_or_dir_path, help='Path to corresponding label file or directory')
    parser.add_argument('fixed', type=ap.file_path, help='Path to atlas file')
    parser.add_argument('fixed_label', type=ap.file_path, help='Path to fixed labels file')
    parser.add_argument('weights_file', type=ap.file_path, help='Load model weights from file')
    parser.add_argument('--save_path', type=ap.dir_path, help='Directory path to save inferred images, DDFs, and labels')
    parser.add_argument("--resize_ratio", type=float, help="Ratio to which the data is resized, e.g., 0.5 with shape (100, 150, 50) -> (50, 75, 25)")
    parser.add_argument("--resize_shape", type=ap.delimited_ints, help="Shape to which the data is resized. May not be exactly this shape, but very similar")
    
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Format data
    data_dicts = dl.format_data(args.moving, args.moving_labels, args.fixed, args.fixed_label)
    
    # Define data loader
    loader = dl.create_dataloader_infer(args, data_dicts)

    # Define device
    device = m.get_device()

    # Define model
    args.lr = 0
    model = m.Model(args, device)
    model.load_weights(args.weights_file)

    # Infer (visualize if a single file)
    from os.path import isfile
    visualize = isfile(args.moving)
    model.infer_val(loader, device, save_path=args.save_path, visualize=visualize)


if __name__ == "__main__":
    main()