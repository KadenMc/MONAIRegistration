import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def select_backend(save_path):
    """
    Selects an interactive or non-interactive matplotlib backend depending on
    whether the figure is being saved to file, i.e., whether path is None.
    
    Parameters:
        save_path (str, None): Figure save path.
    """
    if save_path is None:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')


def show_or_save(save_path):
    """
    Interatively displays an image, or saves it to file, depending on whether a
    path is specified.
    
    Parameters:
        save_path (str, None): Figure save path.
    """
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def visualize_3d(vols, save_path=None):
    """
    Plot slices of a 3D volume by taking a middle slice of each axis.

    Parameters:
        vols (numpy.ndarray, list<numpy.ndarray>): 3D image or list of 3D images.
        save_path (str, None): Figure save path.
    """
    # Select the plotting backend
    select_backend(save_path)

    import neurite as ne
    ne.plot.volume3D(vols)
    show_or_save(save_path)

   
def visualize_deformation(ddf, save_path=None):
    """
    Visualize a deformation field.

    Parameters:
        ddf (numpy.ndarray): A dense deformation field.
        save_path (str, None): Figure save path.
    """
    # Select the plotting backend
    select_backend(save_path)
    
    def meshgridnd_like(in_img, rng_func=range):
        new_shape = list(in_img.shape)
        all_range = [rng_func(i_len) for i_len in new_shape]
        return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])
    
    # Create deformation field visualization
    flow = ddf[1].squeeze()
    DS_FACTOR = 16
    c_xx, c_yy, c_zz = [x.flatten() for x in \
        meshgridnd_like(flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, 0])]
    get_flow = lambda i: flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, i].flatten()
    fig = plt.figure(figsize = (10, 10))
    ax = fig.gca(projection='3d')
    ax.quiver(c_xx, c_yy, c_zz, get_flow(0), get_flow(1), get_flow(2), \
        length=0.9, normalize=True)
    
    # Show if no save path provided, otherwise save
    show_or_save(save_path)


def plot_history(epoch_loss_values, metrics, save_path=None, val_interval=1):
    """
    Plot training history.

    Parameters:
        epoch_loss_values (list<float>): Average batch loss over each epoch.
        metrics (dict of str: list<float>): Average batch metrics every
            val_interval epochs.
        save_path (str, None): Figure save path.
        val_interval (int): Plot validation point every 'val_interval' epochs.
    """
    # Select the plotting backend
    select_backend(save_path)

    num_plots = 1 + len(metrics.keys())

    plt.figure("Training", (6*num_plots, 6))
    plt.subplot(1, 2, 1)
    
    # Plot loss
    plt.title("Epoch Average Loss")
    x = np.arange(len(epoch_loss_values)) + 1
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)

    # Plot metrics
    for i, m in enumerate(metrics):
        plt.subplot(1, 2, i + 2)
        plt.title("Val Mean {}".format(m.capitalize()))
        x = [args.val_interval*i for i in range(len(metric_values))]
        y = metrics[m]
        plt.xlabel("epoch")
        plt.plot(x, y)
    
    # Show if no save path provided, otherwise save
    show_or_save(save_path)


def plot_slice(img, save_path=None, slice=None, slice_depth_percent=0.5, title=None, display=True):
    """
    Returns a given slice in a 3D image, and potentially visualizes it.

    Parameters:
        img (numpy.ndarray): 3D image.
        save_path (str, None): Figure save path.
        slice (int, None): Slice to visualize.
        slice_depth_percent (float, None): Rather than specifying a slice,
            specify approximately how deep into the image to visualize.
        title (str, None): Plot title.
        display (bool): Whether to visualize the slice, or simply return it.
    
    Returns:
        (numpy.ndarray): Image slice.
    """
    assert slice is not None or slice_depth_percent is not None
    
    # Select the plotting backend
    select_backend(save_path)

    if slice is None:
        slice = int(np.round(slice_depth_percent * img.shape[2]))

    if title is None:
        plt.title('Slice {}/{}'.format(slice, img.shape[2]))
    else:
        plt.title(title)
    
    if display:
        plt.imshow(img[:, :, slice], cmap="gray")
        show_or_save(save_path)
    
    return img[:, :, slice]

def plot_slice_overlay(img1, img2, save_path=None, slice=None, slice_depth_percent=0.5, title=None):
    """
    Visualizes two overlayed slices from two 3D images.

    Parameters:
        img1 (numpy.ndarray): 3D image.
        img2 (numpy.ndarray): 3D image.
        save_path (str, None): Figure save path.
        slice (int, None): Slice to visualize.
        slice_depth_percent (float, None): Rather than specifying a slice,
            specify approximately how deep into the image to visualize.
        title (str, None): Plot title.
    """
    # Select the plotting backend
    select_backend(save_path)

    # Get slices
    slice1 = plot_slice(img1, slice=slice, slice_depth_percent=slice_depth_percent, display=False)
    slice2 = plot_slice(img2, slice=slice, slice_depth_percent=slice_depth_percent, display=False)
    assert slice1.shape == slice2.shape
    
    # Create overlay
    overlay = np.zeros((slice1.shape[0], slice1.shape[1], 3), dtype=np.uint8)
    
    scale = lambda x: np.round(((x - x.min())/(x.max() - x.min()))*255).astype(np.uint8)
    
    overlay[:, :, 0] = scale(slice1)
    overlay[:, :, 1] = scale(slice2)
    plt.imshow(overlay)

    # Show if no save path provided, otherwise save
    show_or_save(save_path)
    

def plot_slices(img, n=10, save_path=None):
    """
    Plots a series of slices in a 3D image.

    Parameters:
        img (numpy.ndarray): 3D image.
        n (int): Number of slices to visualize.
        save_path (str, None): Figure save path.
    """
    # Select the plotting backend
    select_backend(save_path)

    from argparsing import join
    # Visualize n equally spaced slices
    for slice in range(n):
        slice = slice * (img.shape[2] // n)
        
        if save_path is None:
            plot_slice(img, save_path=None, slice=slice)
        else:
            plot_slice(img, save_path=join(save_path, 'slice{}'.format(slice) + '.png'), slice=slice)


def check_dataset(files, transforms, verbose=True, save_path=None):
    """
    Plots a set of registration input images.

    Parameters:
        files (list<dict>): MONAI dictionary formatted data.
        transforms (monai.transforms.compose.Compose): MONAI transforms to
            apply - also used to load the data.
        verbose (bool): Whether to print information about the loaded data.
        save_path (str, None): Figure save path.
    """
    # Select the plotting backend
    select_backend(save_path)

    from monai.data import DataLoader, Dataset
    from monai.utils import first
    check_ds = Dataset(data=files, transform=transforms)
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

    # Plot the slice for each image
    slice = moving_image.shape[2]//2
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
    
    # Show if no save path provided, otherwise save
    show_or_save(save_path)


def visualize_inference(moving_image, fixed_image, pred_image, moving_label=None, \
    fixed_label=None, pred_label=None, n=10, save_path=None):
    """
    Plots inputs and predicted registration images.

    Parameters:
        moving_image (numpy.ndarray): Moving image.
        fixed_image (numpy.ndarray): Fixed image.
        pred_image (numpy.ndarray): Predicted/warped image.
        moving_label (numpy.ndarray): Moving label (optional).
        fixed_label (numpy.ndarray): Fixed label (optional).
        pred_label (numpy.ndarray): Predicted/warped label (optional).
        n (int): Number of slices to visualize.
        save_path (str, None): Figure save path.
    """
    # Select the plotting backend
    select_backend(save_path)

    # Visualize n equally spaced slices
    for depth in range(n):
        depth = depth * (moving_image.shape[2] // n)
        plt.figure(depth, (18, 6))

        # Plot moving image
        plt.subplot(1, 6, 1)
        plt.title(f"moving_image d={depth}")
        plt.imshow(moving_image[:, :, depth], cmap="gray")
        
        # Plot predicted image
        plt.subplot(1, 6, 2)
        plt.title(f"pred_image d={depth}")
        plt.imshow(pred_image[:, :, depth], cmap="gray")

        # Plot fixed image
        plt.subplot(1, 6, 3)
        plt.title(f"fixed_image d={depth}")
        plt.imshow(fixed_image[:, :, depth], cmap="gray")
        
        
        # Plot moving label
        if moving_label is not None:
            plt.subplot(1, 6, 4)
            plt.title(f"moving_label d={depth}")
            plt.imshow(moving_label[:, :, depth])
        
        # Plot predicted label
        if pred_label is not None:
            plt.subplot(1, 6, 5)
            plt.title(f"pred_label d={depth}")
            plt.imshow(pred_label[:, :, depth])

        # Plot fixed label
        if fixed_label is not None:
            plt.subplot(1, 6, 6)
            plt.title(f"fixed_label d={depth}")
            plt.imshow(fixed_label[:, :, depth])
        
        # Show if no save path provided, otherwise save
        if save_path is None:
            show_or_save(save_path)
        else:
            show_or_save(save_path[:-4] + '_{}.png'.format(depth))


def visualize_binary_3d(arr, save_path=None):
    """
    Visualize a binary 3D image in 3 dimensions.

    Parameters:
        arr (numpy.ndarray): A binary 3D image.
        save_path (str, None): Figure save path.
    """
    # Select the plotting backend
    select_backend(save_path)

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure
    # Use marching cubes to obtain the surface mesh
    verts, faces, normals, values = measure.marching_cubes(arr, 0)

    # Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces]) # verts[faces] generates a collection of triangles
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, arr.shape[0])
    ax.set_ylim(0, arr.shape[1])
    ax.set_zlim(0, arr.shape[2])
    plt.tight_layout()
    
    # Show if no save path provided, otherwise save
    show_or_save(save_path)
