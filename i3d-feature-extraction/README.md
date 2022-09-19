# feature-extraction

### Installation
1. Open terminal to project base folder
2. python -m venv venv
3. venv\Scripts\activate
4. pip install -r requirements.txt

### Running the Model
    venv\Scripts\activate
    python main.py feature_type=i3d device="cuda:0" on_extraction=save_numpy streams=rgb output_path=./output/rgb file_with_video_paths=./sample/sample_video_paths.txt

### Video Features Documentation
https://iashin.ai/video_features/models/i3d/
