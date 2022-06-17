# Dataloading-code

In addition to the `.py` files that contain the relevant code itself, we also provide two Jupyter-Notebooks to show-case the code.

## General code structure

The code relies on three main packages for dataloading:
* albumentations (augmentations and other transforms)
* pytorch (general dataset structure)
* pycocotools (mscoco-related stuff)

Furthermore, for the compositing of per-fabric datasets into bigger datasets, it uses:
* pytorch-lightning

## Installation

In order to get started, please set-up an appropriate python environment and install all dependencies.
To do this automatically, you can run
```
conda env create -f environment.yml
```

## Running the notebooks

Start-up a local Jupyter server via
```
conda activate olp
jupyter lab
```