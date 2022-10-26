import json
import os
from itertools import groupby

# opens JSON file
json_file = open('../pipeline/data/smarthome_CS_51.json')

# returns JSON object as a dictionary
data = json.load(json_file)

valid_videos = []

# iterates through the json and adds each video id to valid_videos list
for id in data:
    valid_videos.append(id)

# closes JSON file
json_file.close()

extracted_video_ids = []
extraction_video_filenames = []
path = './output/RGB_TEST'

# path = '../pipeline/data/v_iashin_i3d'
files = os.listdir(path)
print(files)

# lists all files in the RGB directory and 
# saves all the video IDs into the extracted_video_ids list 
# and all the video filenames into the extraction_video_filenames list 
for f in files:
    extracted_video_ids.append(f.split('_')[0])
    extraction_video_filenames.append(f.split('_rgb')[0])

# groups the video filenames by the video ID in front of the filename
extraction_video_filenames.sort()
grouped_filenames = [list(i) for j, i in groupby(
    extraction_video_filenames, lambda a: a.split('_')[0]
)]

print(f'Extracted Video: {extracted_videos}')
# displays length of each video list
print(f"current vids: {len(valid_videos)}")
print(f"extracted vids: {len(extracted_video_ids)}")

extracted_set = set(extracted_video_ids)
missing_vids = []

# removes video IDs that do not have RGB npy files
for id in valid_videos:
    if id not in extracted_video_ids:
        data.pop(id)
        missing_vids.append(id)

# iterates through the grouped filenames to retrieve 
# each of the unique variation of the original video ID
# it then creates a new entry based on the variation of the video ID
# in which it duplicates the existing video ID's values
for grouped_filename in grouped_filenames:
    if len(grouped_filename) > 1:
        for filename in grouped_filename:
            video_id = filename.split('_')[0]
            if (video_id in valid_videos) and (filename != video_id):
                data[filename] = data[video_id]

# creates new updated version of smarthome.json with removed video IDs
with open('../pipeline/data/smarthome_CS_51_v2.json', "w") as outfile:
    json.dump(data, outfile)

# print(f"missing vids: {', '.join(missing_vids)}")
print(f"\n{len(missing_vids)} vids removed from smarthome_CS_51_v2.json")
print(f"smarthome_CS_51_v2.json saved to ./pipeline/data/")
