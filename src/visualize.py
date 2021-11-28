import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def select_backend(path):
    """
    Selects an interactive or non-interactive matplotlib backend
    depending on whether the figure is being saved to file.
    
    path: Figure save path
    """
    if path is None:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')


def show_or_save(path):
    """
    Interatively displays an image, or saves it to file,
    depending on whether a path is specified.
    
    path: Figure save path
    """
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def visualize_3d(vols, path=None):
    '''
    Plot slices of a 3D volume by taking a middle slice of each axis.

    vols: A 3d volume or list of 3d volumes
    '''
    import neurite as ne
    select_backend(path)
    ne.plot.volume3D(vols)
    show_or_save(path)

   
def visualize_deformation(pred, path=None):
    '''
    Visualize the deformation field.

    pred: The predicted deformation field
    '''
    
    def meshgridnd_like(in_img, rng_func=range):
        new_shape = list(in_img.shape)
        all_range = [rng_func(i_len) for i_len in new_shape]
        return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])
    
    select_backend(path)
    flow = pred[1].squeeze()
    DS_FACTOR = 16
    c_xx, c_yy, c_zz = [x.flatten() for x in \
        meshgridnd_like(flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, 0])]
    get_flow = lambda i: flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, i].flatten()
    fig = plt.figure(figsize = (10, 10))
    ax = fig.gca(projection='3d')
    ax.quiver(c_xx, c_yy, c_zz, get_flow(0), get_flow(1), get_flow(2), \
        length=0.9, normalize=True)
    show_or_save(path)


def plot_history(args, epoch_loss_values, metric_values, save_path=None):
    select_backend(save_path)
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [args.val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    show_or_save(save_path)


def check_dataset(train_files, train_transforms, verbose=True, save_path=None):
    from monai.data import DataLoader, Dataset
    from monai.utils import first
    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    fixed_image = check_data["fixed_image"][0][0].permute(1, 0, 2)
    moving_image = check_data["moving_image"][0][0].permute(1, 0, 2)
    fixed_label = check_data["fixed_label"][0][0].permute(1, 0, 2)
    moving_label = check_data["moving_label"][0][0].permute(1, 0, 2)
    
    if verbose:
        print(f"fixed_image shape: {fixed_image.shape}, "
            f"moving_image shape: {moving_image.shape}")
        print(f"fixed_label shape: {fixed_label.shape}, "
            f"moving_label shape: {moving_label.shape}")

    # Plot the slice [:, :, slice]
    slice = moving_image.shape[2]//2
    select_backend(save_path)
    plt.figure("check", (12, 6))
    plt.subplot(1, 4, 1)
    plt.title("moving_image")
    plt.imshow(moving_image[:, :, slice], cmap="gray")
    plt.subplot(1, 4, 2)
    plt.title("moving_label")
    plt.imshow(moving_label[:, :, slice])
    plt.subplot(1, 4, 3)
    plt.title("fixed_image")
    plt.imshow(fixed_image[:, :, slice], cmap="gray")          
    plt.subplot(1, 4, 4)
    plt.title("fixed_label")
    plt.imshow(fixed_label[:, :, slice])
    
    show_or_save(save_path)


def visualize_inference(moving_image, moving_label, fixed_image, fixed_label, pred_image, pred_label, \
    visualize_n=10, save_path=None):
    # Visualize visualize_n equally spaced slices
    select_backend(save_path)
    for depth in range(visualize_n):
        depth = depth * (moving_image.shape[2] // visualize_n)
        # plot the slice [:, :, 80]
        plt.figure(depth, (18, 6))
        plt.subplot(1, 6, 1)
        plt.title(f"moving_image d={depth}")
        plt.imshow(moving_image[:, :, depth], cmap="gray")
        plt.subplot(1, 6, 2)
        plt.title(f"moving_label d={depth}")
        plt.imshow(moving_label[:, :, depth])
        plt.subplot(1, 6, 3)
        plt.title(f"fixed_image d={depth}")
        plt.imshow(fixed_image[:, :, depth], cmap="gray")
        plt.subplot(1, 6, 4)
        plt.title(f"fixed_label d={depth}")
        plt.imshow(fixed_label[:, :, depth])
        plt.subplot(1, 6, 5)
        plt.title(f"pred_image d={depth}")
        plt.imshow(pred_image[:, :, depth], cmap="gray")
        plt.subplot(1, 6, 6)
        plt.title(f"pred_label d={depth}")
        plt.imshow(pred_label[:, :, depth])
        if save_path is None:
            show_or_save(save_path)
        else:
            show_or_save(save_path[:-4] + '_{}.png'.format(depth))