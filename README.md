# ict3104-team05-2022

## Team Members
- Eddie Tan (2002226)
- Nicholas Bingei (2002983)
- Ter Kai Siang (2001476)
- Jessica Tan Guang Hui (2002253)
- Muhammad Muhaimin Bin Mohd Ikhsan (2002950)
- Jimeno Johanna Charissa Mortel (2002171)

## HOI Hub

This project is currently preloaded with these models:

- [Toyota Smarthome (TSU) project](https://github.com/dairui01/Toyota_Smarthome)
- [I3D Feature Extraction](https://github.com/v-iashin/video_features)
- [TSU evaluation](https://github.com/dairui01/TSU_evaluation)
- [Nvidia STEP: Spatio-Temporal Progressive Learning for Video Action Detection](https://github.com/NVlabs/STEP)
- [MS-TCT: Multi-Scale Temporal ConvTransformer for Action Detection]()
- [Pytorch I3D models trained on Kinetics](https://github.com/piergiaj/pytorch-i3d)

## Installation for windows
Ensure that you have the following installed:
- [Anaconda](https://www.anaconda.com/products/distribution)
- Create a virtual environment in conda using the following command:
```
 conda env create -f env.yml
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
- Install the following packages using pip:
```
pip install --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt
```
- Run the following command to start the juypter notebook:
```
jupyter notebook
```

## Installation for Linux
Ensure that you have the following installed:
- [Anaconda](https://www.anaconda.com/products/distribution)
- Create a virtual environment in conda using the following command:
```
 conda env create -f env_linux.yml
```
- Activate the virtual environment and install torch using the following command:
```
 conda activate ict3104
 conda install pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 -c pytorch -c nvidia
```
- Create and activate a virtualenv
```
pip install virtualenv
python -m venv venv
source venv/bin/activate
```
- Install the following packages using pip:
```
pip install -r requirements_linux.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
- Run the following command to start the juypter notebook:
```
jupyter notebook
```
## Dataset & Video
Place all your dataset folder into ```data/dataset/``` folder
Place all your video into ```data/video/``` folder

## Environment File for Wandb

- Create a .env in the project root directory
- Append the following to the .env file:

```
WANDB_API_KEY=<your_wandb_api_key>
```