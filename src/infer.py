import torch

# Local imports
import argparsing as ap
import model as m
from dataloader import create_dataloader_infer


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


def main():
    # Parse arguments
    args = ap.parse_arguments_infer()
    
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
    device = get_device()

    # Define model
    args.lr = 0
    model = m.Model(args, device)
    model.load_weights(args.weights_file)

    # Infer
    model.infer_val(loader, device)


if __name__ == "__main__":
    main()