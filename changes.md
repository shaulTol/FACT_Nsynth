# FACT Repository Setup and Optimization Guide

This guide provides detailed steps for setting up and optimizing the FACT repository .
It includes solutions for common issues we encountered.

## Prerequisites
- Conda
- Python 3.6

## 1. Initial Repository Setup

### Clone the Repository
```bash
git clone https://github.com/MediaBrain-SJTU/FACT.git
cd FACT
```

### Download Dataset
1. Download PACS dataset from [Google Drive](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ)
2. Create the following directory structure:
```
FACT/
└── data/
    └── images/
        └── PACS/
            └── kfold/
                ├── art_painting/
                ├── cartoon/
                ├── photo/
                └── sketch/
```

## 2. Data Preparation

### Dataset Splits
1. Download split files from [Google Drive](https://drive.google.com/drive/folders/1i23DCs4TJ8LQsmBiMxsxo6qZsbhiX0gw)
2. Modify filenames by removing "_kfold" suffix:
   - Change `art_painting_test_kfold.txt` to `art_painting_test.txt`
   - Do the same for all domains and splits (train/val/test)

### File Path Adjustment
1. Update file paths in txt files to match your system path:
```bash
# Run this command in the directory containing the txt files
sed -i '' 's|^|/YOUR/PATH/TO/FACT/data/images/PACS/kfold/|' *.txt
```

### Fix Class Labels
The original files use 1-based indexing, but the model expects 0-based indexing. Fix this with:
```bash
for file in *.txt; do
    awk '{print $1, $2-1}' "$file" > tmp && mv tmp "$file"
done
```

## 3. Environment Setup

### Create and Activate Conda Environment
```bash
conda create --name fact_env python=3.6
conda activate fact_env
```

### Install PyTorch and Dependencies
1. Download PyTorch 1.1.0 wheel from [PyTorch website](https://download.pytorch.org/whl/cpu/torch_stable.html)
   - For macOS: `torch-1.1.0-cp36-none-macosx_10_7_x86_64.whl`
   - For Windows: Choose appropriate Windows wheel

2. Install dependencies:
```bash
cd FACT
pip install torch-1.1.0-cp36-none-macosx_10_7_x86_64.whl  # or your downloaded wheel
pip install torchvision==0.3.0
conda install opencv
pip install tensorflow
pip install scipy
```


## 4. Directory Setup
Create necessary directories:
```bash
mkdir ckpt  # for checkpoints
mkdir train_logs  # for training logs
```

Update paths in shell_train.py:
```python
input_dir = '/YOUR/PATH/TO/FACT/data/datalists'
output_dir = '/YOUR/PATH/TO/FACT/train_logs'
```

## 5. Performance Optimization for Training

### Create ResNet18 Configuration
a. In the config folder, create `ResNet18.py` (duplicate of ResNet50.py) with these modifications:

```python
encoder = {
    "name": "resnet18",
}
networks["encoder"] = encoder

classifier = {
    "name": "base",
    "in_dim": 512,  # Changed from 2048 (ResNet50) to 512 (ResNet18)
    "num_classes": num_classes,
    "cls_type": "linear"
}
```

b. Update configuration in shell files:
   - In `shell_train.py`: Change `config = "PACS/ResNet50"` to `config = "PACS/ResNet18"`
   - In `shell_test.py`: Make the same change



## 6. Running Training

Start training with:
```bash
python shell_train.py -d=art_painting
```
Use the `-d` argument to specify the held-out target domain (art_painting, cartoon, photo, or sketch).



## References

Original Paper:
```
@InProceedings{Xu_2021_CVPR,
    author    = {Xu, Qinwei and Zhang, Ruipeng and Zhang, Ya and Wang, Yanfeng and Tian, Qi},
    title     = {A Fourier-Based Framework for Domain Generalization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {14383-14392}
}
```
