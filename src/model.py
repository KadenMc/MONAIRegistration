import os
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from monai.losses import BendingEnergyLoss, MultiScaleLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet

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

class Model:
    def __init__(self, args, device):
        self.model = LocalNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            num_channel_initial=32,
            extract_levels=[0, 1, 2, 3],
            out_activation=None,
            out_kernel_initializer="zeros").to(device)
        self.warp_layer = Warp().to(device)
        self.image_loss = MSELoss()
        self.label_loss = DiceLoss()
        self.label_loss = MultiScaleLoss(self.label_loss, scales=[0, 1, 2, 4, 8, 16])
        self.regularization = BendingEnergyLoss()
        self.optimizer = Adam(self.model.parameters(), args.lr)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    def load_weights(self, file):
        self.model.load_state_dict(torch.load(file))
    
    def forward(self, batch_data, device):
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
    
    
    def train(self, args, train_loader, val_loader, device, model_path):
        from argparsing import join
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []

        for epoch in range(args.max_epochs):
            # Perform validation
            if (epoch + 1) % args.val_interval == 0 or epoch == 0:
                self.model.eval()
                with torch.no_grad():
                    for val_data in val_loader:

                        val_ddf, val_pred_image, val_pred_label = self.forward(
                            val_data, device)

                        val_fixed_image = val_data["fixed_image"].to(device)
                        val_fixed_label = val_data["fixed_label"].to(device)
                        self.dice_metric(y_pred=val_pred_label, y=val_fixed_label)

                    metric = self.dice_metric.aggregate().item()
                    self.dice_metric.reset()
                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(self.model.state_dict(), args.save_weights_file)
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} "
                        f"current mean dice: {metric:.4f}\n"
                        f"best mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )
            
            # Perform training
            print("-" * 10)
            print(f"epoch {epoch + 1}/{args.max_epochs}")
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
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


        print(f"train completed, "
            f"best_metric: {best_metric:.4f}  "
            f"at epoch: {best_metric_epoch}")

        return epoch_loss_values, metric_values
    
    def infer_val(self, loader, device, save_path=None, visualize=True, visualize_n=10, visualize_save_path=None):
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
                    from argparsing import join
                    from dataloader import check_extensions, save_nii_file
                    from numpy import float64
                    
                    # Extract filename information
                    filename = os.path.basename(os.path.normpath( \
                        data['moving_image_meta_dict']['filename_or_obj'][0]))
                    ext = check_extensions(filename)
                    
                    # Save files predicted image, deformation field, and label
                    save_nii_file(join(save_path, filename[:-len(ext)] + '_ddf.nii.gz'), ddf.astype(float64))
                    save_nii_file(join(save_path, filename[:-len(ext)] + '_pred.nii.gz'), pred_image.astype(float64))

                    # Only saves if a moving label was provided
                    if pred_label is not None:
                        save_nii_file(join(save_path, filename[:-len(ext)] + '_labels.nii.gz'), pred_label.astype(float64))
                    