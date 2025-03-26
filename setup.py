from pathlib import Path

from setuptools import find_packages, setup

NAME = "remfx"
DESCRIPTION = "Universal audio effect removal"
URL = "https://github.com/mhrice/RemFx"
EMAIL = "m.rice@se22.qmul.ac.uk"
AUTHOR = "Matthew Rice"
REQUIRES_PYTHON = ">=3.12.0"
VERSION = "0.0.1"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.1",
        "torchaudio>=2.5.1",
        "scipy",
        "numpy",
        "torchvision",
        "lightning",
        "einops",
        "hydra-core",
        "auraloss",
        "pyloudnorm",
        "pedalboard",
        "librosa",
        "torchmetrics",
        "tensorboard",
        "pandas",
    ],
    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
)
