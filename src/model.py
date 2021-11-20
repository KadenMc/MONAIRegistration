import os
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from monai.losses import BendingEnergyLoss, MultiScaleLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet

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

        # predict DDF through LocalNet
        ddf = self.model(torch.cat((moving_image, fixed_image), dim=1))

        # warp moving image and label with the predicted ddf
        pred_image = self.warp_layer(moving_image, ddf)
        pred_label = self.warp_layer(moving_label, ddf)

        return ddf, pred_image, pred_label
    
    
    def train(self, args, train_loader, val_loader, device, model_path):
        from argparsing import join
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []

        for epoch in range(args.max_epochs):
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
    
    def infer(self, val_data, device, visualize=True, visualize_n=10, visualize_save_path=None):
        self.model.eval()
        with torch.no_grad():
            # Deformation field, image, label
            val_ddf, val_pred_image, val_pred_label = self.forward(val_data, device)
            val_pred_image = val_pred_image.cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_pred_label = val_pred_label.cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_moving_image = val_data["moving_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_moving_label = val_data["moving_label"].cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_fixed_image = val_data["fixed_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_fixed_label = val_data["fixed_label"].cpu().numpy()[0, 0].transpose((1, 0, 2))

            if visualize:
                from visualize import visualize_inference
                visualize_inference(val_moving_image, val_moving_label, val_fixed_image, \
                    val_fixed_label, val_pred_image, val_pred_label, \
                    visualize_n=visualize_n, save_path=visualize_save_path)
    
        return val_ddf, val_moving_image, val_moving_label, val_fixed_image, val_fixed_label, \
            val_pred_image, val_pred_label
    
    def infer_val(self, val_loader, device, visualize=True, visualize_n=10, visualize_save_path=None):
        # Inference using pretrained weights, visualize the result at different depth
        self.model.eval()
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                # Deformation field, image, label
                images = self.infer(val_data, device, visualize=visualize, visualize_n=visualize_n, \
                    visualize_save_path=visualize_save_path)