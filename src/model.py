import os
import torch
from torch.nn import MSELoss
from monai.losses import BendingEnergyLoss, MultiScaleLoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MSEMetric
from monai.losses import LocalNormalizedCrossCorrelationLoss, GlobalMutualInformationLoss
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


class LRScheduler():
    """
    Learning rate scheduler. If a given metric does not increase/decrease for 
    'patience' epochs, then decrease the learning rate by some 'factor'.
    
    new_lr = old_lr * factor
    """
    def __init__(self, optimizer, patience=10, min_lr=1e-6, factor=0.8, \
        mode='min', verbose=True):
        """
        Parameters:
            optimizer: The optimizer.
            patience (int): Number of epochs to wait before updating lr.
            min_lr (float): Minimum lr value.
            factor (float): Percent decrease on each update.
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.mode = mode
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( \
            self.optimizer, mode=self.mode, patience=self.patience, \
            factor=self.factor, min_lr=self.min_lr, verbose=verbose)
    
    
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, mode="min"):
        """
        Parameters:
            patience (int): Number of epochs to wait with no metric improvement
                before early stopping.
            min_delta (float): Minimum difference between old and new metric to
                be considered as an improvement
        """
        assert mode == "min" or mode == "max"
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
    
    
    def __call__(self, metric):
        # If this is the first epoch
        if self.best_metric == None:
            self.best_metric = metric
        
        else:
            # If wanting the metric to decrease
            if self.mode == "min":
                if self.best_metric - metric >= self.min_delta:
                    # Reset counter if metric improves
                    self.best_metric = metric
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print('Stopping early...')
                        return True
            
            # If wanting the metric to increase
            elif self.mode == "max":
                if metric - self.best_metric >= self.min_delta:
                    # Reset counter if metric improves
                    self.best_metric = metric
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print('Stopping early.')
                        return True
        
        return False

class History:
    """
    A container class to store the training loss and validation set evaluation
    metrics and allow for easy saving and loading with pickle.

    Attributes
    ----------
    losses (list<float>) : Average training loss over each epoch.
    metrics (dict of str: list<float>) : Average validation metrics every
        self.val_interval epochs.
    """

    def __init__(self, losses, metrics):
        self.losses = losses
        self.metrics = metrics



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
    mse_metric : Mean squared error metric function
    ncc_metric : Normalized cross correlation metric function
    mi_metric = Mutua information metric function
    dice_metric : Dice metric function
    hausdorff_metric : Hausdorff distance metric function


    Methods
    -------
    load_weights(file):
        Loads weights into the model from a weights file.
    
    forward(batch_data, device):
        Sends input data into the model and returns the output. Labels must be provided.

    forward_val(batch_data, device):
        Sends input data into the model and returns the output. Labels may be provided optionally.
    
    train(train_loader, val_loader, device, max_epochs, save_weights_file=None):
        Trains the model and may perform validation at the end of each epoch.
    
    infer(loader, device, save_path=None, visualize=True, visualize_n=10, visualize_save_path=None):
        Infers outputs from the input data, visualizing and saving the outputs accordingly.
    """

    def __init__(self, device, val_interval=1, lr=1e-5, lr_factor=0.5, lr_patience=5, es_patience=10):
        """
        Model initialization.

        Parameters:
            device (torch.device): Device on which to train.
            val_interval (int): Perform validation every 'val_interval' epochs.
            lr (float): Learning rate.
            lr_factor (float): Learning rate percentage decrease factor.
            lr_patience (int): Number of epochs between checking to decrease learning rate.
            es_patience (int): Number of epochs between early stopping checks.
        """
        self.model = LocalNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            num_channel_initial=32,
            extract_levels=[0, 1, 2, 3],
            out_activation=None,
            out_kernel_initializer="zeros").to(device)

        # Deformation function/layer
        self.warp_layer = Warp().to(device)
        
        # Losses
        self.image_loss = MSELoss()
        self.label_loss = DiceLoss()
        self.label_loss = MultiScaleLoss(self.label_loss, scales=[0, 1, 2, 4, 8, 16])
        self.regularization = BendingEnergyLoss()

        self.val_interval = val_interval

        # Optimization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        
        # Learning rate scheduler
        self.lr_scheduler = LRScheduler(self.optimizer, factor=lr_factor, \
            patience=lr_patience // val_interval)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=es_patience // val_interval)

        # Metrics
        self.mse_metric = MSEMetric()
        self.ncc_metric = LocalNormalizedCrossCorrelationLoss(spatial_dims=3)
        self.mi_metric = GlobalMutualInformationLoss()
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    
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
        save_weights_file=None):
        """
        Trains the model and may perform validation at the end of each epoch.

        Parameters:
            train_loader (monai.data.DataLoader): Training dataloader.
            val_loader (monai.data.DataLoader): Validation dataloader.
            device (torch.device): Device on which to train.
            max_epochs (int): Maximum number of epochs.
            save_weights_file (str): File to save the model weights.
        
        Returns:
            losses (list<float>): Average training loss over each epoch.
            metrics (dict of str: list<float>): Average validation metrics
                every self.val_interval epochs.
        """
        # Define metric & tracking variables
        best_dice = -1
        best_dice_epoch = -1
        losses = []
        val_losses = []

        mse_metric_values = []
        ncc_metric_values = []
        mi_metric_values = []
        dice_metric_values = []
        hausdorff_metric_values = []

        # Training loop
        for epoch in range(max_epochs):

            # Perform validation
            if epoch % self.val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    epoch_val_loss = 0
                    ncc = 0
                    mi = 0
                    step = 0
                    for val_data in val_loader:
                        step += 1
                        
                        val_ddf, val_pred_image, val_pred_label = self.forward(val_data, device)

                        # Send to device
                        val_fixed_image = val_data["fixed_image"].to(device)
                        val_fixed_label = val_data["fixed_label"].to(device)

                        # Get loss
                        val_loss = self.image_loss(val_pred_image, val_fixed_image) + 100 * \
                            self.label_loss(val_pred_label, val_fixed_label) + 10 * self.regularization(val_ddf)

                        epoch_val_loss += val_loss.item()

                        # Get image metrics
                        self.mse_metric(y_pred=val_pred_image, y=val_fixed_image)
                        ncc += self.ncc_metric(val_pred_image, val_fixed_image).item()
                        mi += self.mi_metric(val_pred_image, val_fixed_image).item()

                        # Get label metrics
                        self.dice_metric(y_pred=val_pred_label.byte(), y=val_fixed_label.byte())
                        self.hausdorff_metric(y_pred=val_pred_label.byte(), y=val_fixed_label.byte())

                    epoch_val_loss /= step
                    val_losses.append(epoch_val_loss)
                    print(f"Epoch {epoch} average validation loss: {epoch_val_loss:.4f}")

                    # Record and reset validation metrics
                    mse = self.mse_metric.aggregate().item()
                    self.mse_metric.reset()
                    mse_metric_values.append(mse)

                    ncc_metric_values.append(-(ncc/step))
                    mi_metric_values.append(-(mi/step))

                    dice = self.dice_metric.aggregate().item()
                    self.dice_metric.reset()
                    dice_metric_values.append(dice)

                    hausdorff = self.hausdorff_metric.aggregate().item()
                    self.hausdorff_metric.reset()
                    hausdorff_metric_values.append(hausdorff)

                    if dice > best_dice:
                        best_dice = dice
                        best_dice_epoch = epoch

                        if save_weights_file is not None:
                            torch.save(self.model.state_dict(), save_weights_file)
                            print("Saved new best metric model")
                    print(
                        f"Epoch: {epoch} "
                        f"Mean MSE: {mse_metric_values[-1]:.4f}\n"
                        f"Mean NCC: {ncc_metric_values[-1]:.4f}\n"
                        f"Mean MI: {mi_metric_values[-1]:.4f}\n"
                        f"Mean Dice: {dice_metric_values[-1]:.4f}\n"
                        f"Mean Hausdorff: {hausdorff_metric_values[-1]:.4f}\n"
                        f"Best mean dice: {best_dice:.4f} "
                        f"at epoch: {best_dice_epoch}"
                    )

                    # Update learning rate
                    self.lr_scheduler(epoch_val_loss)
                    
                    # Check for early stopping
                    if self.early_stopping(epoch_val_loss):
                        break
            
            # Perform training
            print("-" * 10)
            print(f"Epoch {epoch + 1}/{max_epochs}")
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
            losses.append(epoch_loss)
            print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Give end of training information
        print(f"Train completed, "
            f"Best_metric: {best_dice:.4f}  "
            f"at epoch: {best_dice_epoch}")

        # Prepare metric dictionary
        metrics = dict()
        metrics['mse'] = mse_metric_values
        metrics['ncc'] = ncc_metric_values
        metrics['mi'] = mi_metric_values
        metrics['dice'] = dice_metric_values
        metrics['hausdorff'] = hausdorff_metric_values
        metrics['val loss'] = val_losses

        history = History(losses, metrics)
        return history
    

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

        # If saving, create the directories for the different saved output types
        if save_path is not None:
            ddf_path = join(save_path, 'ddf')
            if not os.path.isdir(ddf_path):
                os.mkdir(ddf_path)

            pred_path = join(save_path, 'pred')
            if not os.path.isdir(pred_path):
                os.mkdir(pred_path)

            label_path = join(save_path, 'labels')
            if not os.path.isdir(label_path):
                os.mkdir(label_path)


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
                    save_nii_file(join(ddf_path, filename[:-len(ext)] + '.nii.gz'), \
                        ddf.astype(float64))
                    save_nii_file(join(pred_path, filename[:-len(ext)] + '.nii.gz'), \
                        pred_image.astype(float64))

                    # Only saves if a moving label was provided
                    if pred_label is not None:
                        save_nii_file(join(label_path, filename[:-len(ext)] + '.nii.gz'), \
                            pred_label.astype(float64))
