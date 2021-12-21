import os
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from monai.losses import BendingEnergyLoss, MultiScaleLoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MSEMetric
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet

# Local imports
from argparsing import join


def get_device(verbose=True):
        """
        Get the device on which to train. Use a GPU if possible, otherwise CPU.

        Parameters:
            verbose (bool): Specify whether to print information about device & device selection.
        
        Returns:
            device (torch.device): A torch device.
        """
        print("torch.cuda.is_available()", torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if verbose:
            if device.type == 'cuda':
                print("Using Device:", torch.cuda.get_device_name(0))
            else:
                print("Using Device:", device)
        
        return device

class Model:
    """
    A class handling the model components and functionality.

    Attributes
    ----------
    model : Model object
    warp_layer : Warping function
    image_loss : Image based loss function
    label_loss : Label (feature based) loss function
    regularization : Deformation loss function
    optimizer : Training optimizer
    dice_metric : Dice metric function


    Methods
    -------
    load_weights(file):
        Loads weights into the model from a weights file.
    
    forward(batch_data, device):
        Sends input data into the model and returns the output.

    forward_val(batch_data, device):
        Sends input data into the model and returns the output. Labels may be provided optionally.
    
    train(train_loader, val_loader, device, max_epochs, save_weights_file=None, val_interval=1):
        Trains the model and may perform validation at the end of each epoch.
    
    infer(loader, device, save_path=None, visualize=True, visualize_n=10, visualize_save_path=None):
        Infers outputs from the input data, visualizing and saving the outputs accordingly.
    """

    def __init__(self, device, lr=0):
        """
        Model initialization.

        Parameters:
            device (torch.device): Device on which to train.
            lr (float): Learning rate.
        """
        self.model = LocalNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            num_channel_initial=32,
            extract_levels=[0, 1, 2, 3],
            out_activation=None,
            out_kernel_initializer="zeros").to(device)

        # Deformation
        self.warp_layer = Warp().to(device)
        
        # Losses
        self.image_loss = MSELoss()
        self.label_loss = DiceLoss()
        self.label_loss = MultiScaleLoss(self.label_loss, scales=[0, 1, 2, 4, 8, 16])
        self.regularization = BendingEnergyLoss()

        # Optimization
        self.optimizer = Adam(self.model.parameters(), lr)

        # Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.hausdorff_metric = HausdorffDistanceMetric()
        self.mse_metric = MSEMetric()
    
    def load_weights(self, file):
        """
        Load model weights from a provided file.

        Parameters:
            file (str): File from which to load the weights.
        """
        self.model.load_state_dict(torch.load(file))
    
    def forward(self, batch_data, device):
        """
        Sends input data into the model and returns the output.

        Parameters:
            batch_data (list<dict>): Training batch size in MONAI's dictionary formatting.
            device (torch.device): Device to which the data is sent.
        
        Returns:
            ddf (torch.FloatTensor): Dense deformation field.
            pred_image (torch.FloatTensor): Warped moving image.
            pred_label (torch.FloatTensor): Warped moving image label.
        """
        fixed_image = batch_data["fixed_image"].to(device)
        moving_image = batch_data["moving_image"].to(device)
        moving_label = batch_data["moving_label"].to(device)

        # Predict DDF through LocalNet
        ddf = self.model(torch.cat((moving_image, fixed_image), dim=1))

        # Warp moving image and label with the predicted DDF
        pred_image = self.warp_layer(moving_image, ddf)
        pred_label = self.warp_layer(moving_label, ddf)

        return ddf, pred_image, pred_label
    
    def forward_val(self, batch_data, device):
        """
        Sends input data into the model and returns the output. Labels may be provided optionally.

        Parameters:
            batch_data (list<dict>): Training batch size in MONAI's dictionary formatting.
            device (torch.device): Device to which the data is sent.

        Returns:
            ddf (torch.FloatTensor): Dense deformation field.
            pred_image (torch.FloatTensor): Warped moving image.
            pred_label (torch.FloatTensor, None): A warped moving image label if a moving label
                is provided, otherwise None.
        """
        # If a moving label is provided, return a warped label prediction
        # Otherwise, return only the DDF and warped image
        if "moving_label" in batch_data:
            return self.forward(batch_data, device)
        
        fixed_image = batch_data["fixed_image"].to(device)
        moving_image = batch_data["moving_image"].to(device)

        # Predict DDF through LocalNet
        ddf = self.model(torch.cat((moving_image, fixed_image), dim=1))

        # Warp moving image with the predicted DDF
        pred_image = self.warp_layer(moving_image, ddf)
        return ddf, pred_image, None
    
    
    def train(self, train_loader, val_loader, device, max_epochs, \
        save_weights_file=None, val_interval=1):
        """
        Trains the model and may perform validation at the end of each epoch.

        Parameters:
            train_loader (monai.data.DataLoader): Training dataloader.
            val_loader (monai.data.DataLoader): Validation dataloader.
            device (torch.device): Device on which to train.
            max_epochs (int): Maximum number of epochs.
            save_weights_file (str): File to save the model weights.
            val_interval (int): Perform validation every 'val_interval' epochs.
        
        Returns:
            epoch_loss_values (list<float>): Average batch loss over each epoch.
            metrics (dict of str: list<float>): Average batch metrics every
                val_interval epochs.
        """
        # Define metric & tracking variables
        best_dice = -1
        best_dice_epoch = -1
        epoch_loss_values = []
        dice_metric_values = []
        hausdorff_metric_values = []
        mse_metric_values = []

        # Training loop
        for epoch in range(max_epochs):

            # Perform validation
            if epoch % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_ddf, val_pred_image, val_pred_label = self.forward(val_data, device)

                        # Send to device
                        val_fixed_image = val_data["fixed_image"].to(device)
                        val_fixed_label = val_data["fixed_label"].to(device)

                        # Get metrics
                        self.dice_metric(y_pred=val_pred_label, y=val_fixed_label)
                        self.hausdorff_metric(y_pred=val_pred_label, y=val_fixed_label)
                        self.mse_metric(y_pred=val_pred_label, y=val_fixed_label)

                    # Record and reset validation metrics
                    dice = self.dice_metric.aggregate().item()
                    self.dice_metric.reset()
                    dice_metric_values.append(dice)

                    hausdorff = self.hausdorff_metric.aggregate().item()
                    self.hausdorff_metric.reset()
                    hausdorff_metric_values.append(hausdorff)

                    mse = self.mse_metric.aggregate().item()
                    self.mse_metric.reset()
                    mse_metric_values.append(mse)

                    if dice > best_dice:
                        best_dice = dice
                        best_dice_epoch = epoch

                        if save_weights_file is not None:
                            torch.save(self.model.state_dict(), save_weights_file)
                            print("Saved new best metric model")
                    print(
                        f"Current epoch: {epoch} "
                        f"Current mean dice: {dice:.4f}\n"
                        f"Current mean hausdorff: {hausdorff:.4f}\n"
                        f"Current mean mse: {mse:.4f}\n"
                        f"Best mean dice: {best_dice:.4f} "
                        f"at epoch: {best_dice_epoch}"
                    )
            
            # Perform training
            print("-" * 10)
            print(f"Epoch {epoch}/{max_epochs}")
            self.model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                self.optimizer.zero_grad()

                ddf, pred_image, pred_label = self.forward(batch_data, device)

                fixed_image = batch_data["fixed_image"].to(device)
                fixed_label = batch_data["fixed_label"].to(device)
                
                loss = self.image_loss(pred_image, fixed_image) + 100 * \
                    self.label_loss(pred_label, fixed_label) + 10 * self.regularization(ddf)
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"Epoch {epoch} average loss: {epoch_loss:.4f}")

        # Give end of epoch information
        print(f"Train completed, "
            f"Best_metric: {best_metric:.4f}  "
            f"at epoch: {best_metric_epoch}")

        # Prepare metric dictionary
        metrics = dict()
        metrics['dice'] = dice_metric_values
        metrics['hausdorff'] = hausdorff_metric_values
        metrics['mse'] = mse_metric_values

        return epoch_loss_values, metrics
    

    def infer(self, loader, device, save_path=None, visualize=True, \
        visualize_n=10, visualize_save_path=None):
        """
        Infers outputs from the input data, visualizing and saving the outputs accordingly.

        Parameters:
            loader (monai.data.DataLoader): A dataloader.
            device (torch.device): Device on which to perform inference.
            save_path (str, None): Path to which predicted images will be saved.
                They will not be saved if None.
            visualize (bool): Whether to visualized the inputs and outputs.
            visualize_n (int): How many slices to visualize from the inputs and outputs.
            visualize_save_path (str, None): Where to save the visualizations.
                They will not be saved if None.
        """
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                # Get deformation field, image, label
                ddf, pred_image, pred_label = self.forward_val(data, device)

                # Prepare images for visualization and saving
                moving_image = data["moving_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
                fixed_image = data["fixed_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
                ddf = ddf.cpu().numpy()[0, 0].transpose((1, 0, 2))
                pred_image = pred_image.cpu().numpy()[0, 0].transpose((1, 0, 2))

                # If labels were provided
                if pred_label is not None:
                    pred_label = pred_label.cpu().numpy()[0, 0].transpose((1, 0, 2))
                    moving_label = data["moving_label"].cpu().numpy()[0, 0].transpose((1, 0, 2))
                    fixed_label = data["fixed_label"].cpu().numpy()[0, 0].transpose((1, 0, 2))
                else:
                    pred_label = None
                    moving_label = None
                    fixed_label = None
                
                # Visualize
                if visualize:
                    from visualize import visualize_inference
                    visualize_inference(moving_image, fixed_image, pred_image, \
                        moving_label=moving_label, fixed_label=fixed_label, pred_label=pred_label, \
                        n=visualize_n, save_path=visualize_save_path)
                
                # Save the files
                if save_path is not None:
                    from dataloader import get_recognized_extension, save_nii_file
                    from numpy import float64
                    
                    # Extract filename information
                    filename = os.path.basename(os.path.normpath( \
                        data['moving_image_meta_dict']['filename_or_obj'][0]))
                    ext = get_recognized_extension(filename)
                    
                    # Save files predicted image, deformation field, and label
                    save_nii_file(join(save_path, filename[:-len(ext)] + '_ddf.nii.gz'), \
                        ddf.astype(float64))
                    save_nii_file(join(save_path, filename[:-len(ext)] + '_pred.nii.gz'), \
                        pred_image.astype(float64))

                    # Only saves if a moving label was provided
                    if pred_label is not None:
                        save_nii_file(join(save_path, filename[:-len(ext)] + '_labels.nii.gz'), \
                            pred_label.astype(float64))
        