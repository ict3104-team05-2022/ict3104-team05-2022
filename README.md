# ict3104-team05-2022
ICT3104 Team05 2
Nvidia Project

## Installation
Ensure that you have the following installed:
- [Anaconda](https://www.anaconda.com/products/distribution)
- Create a virtual environment in conda using the following command:
```
 conda env create -n ict3104 --file env.yml
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
## Environment File
- Create a .env in the project root directory
- Append the following to the .env file:
```
WANDB_API_KEY=<your_wandb_api_key>
```