from setuptools import setup, find_packages

setup(
    name="yolosam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision", 
        "numpy",
        "Pillow",
        "albumentations",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "wandb",
        "monai",
        "tqdm",
        "ultralytics",
        "nnunetv2",
    ],
    python_requires=">=3.8",
)