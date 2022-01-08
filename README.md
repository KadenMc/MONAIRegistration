# MONAIRegistration

An unsupervised, deep-learning pipeline for deformable, atlas-based, pairwise image registration, utilizing the MONAI library.

We formulate registration as a function that maps an input image pair to a deformation field which aligns these images. It is assumed that all moving images are registered to some constant fixed image, i.e., an atlas.


&nbsp;


## Files &amp; Usage


Files with command-line arguments &amp; related **--help** functionality:

- `main.py`: Train a registration model.
- `infer.py`: Infer &amp; perform registration with a trained model.
- `get_metrics.py`: Get metrics from a directory of moving images and (optionally) labels.
- `create_labels.py`: Create binary region of interest labels.
- `preprocessing.py`: Preprocess a file or directory of files.
- `affine.py`: Perform affine registration.

Supporting files:
- `argparsing.py`: Generalized functioning required for command-line argparsing.
- `dataloader.py`: Data formatting, loading, saving, and transforms.
- `model.py`: Model, training, and inference functionality.
- `resampling.py`: Image resampling.
- `visualize.py`: Visualization functionality.