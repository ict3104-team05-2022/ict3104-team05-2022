# ict3104-team05-2022
ICT3104 Team05 2
Nvidia Project

## Installation
Ensure that you have the following installed:
- [Anaconda](https://www.anaconda.com/products/distribution)
- Create a virtual environment using the following command:
```
 conda env create -n ict3104 --file environment.yml
```
- Activate the virtual environment using the following command:
```
 conda activate ict3104
```
- Create and activate a virtualenv
```
    pip install virtualenv
    python -m venv venv
    venv\Scripts\activate
```
- Install the following packages:
```
 pip install -r requirements.txt
 pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
- Run the following command to start the juypter notebook:
```
 jupyter notebook
```
## Installation V2
Run the setup.ipynb notebook to install the dependencies.

## Binder
You can access the binder here
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ict3104-team05-2022/ict3104-team05-2022/dev)
