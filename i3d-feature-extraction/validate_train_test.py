import json
import os

# opens JSON file
json_file = open('../pipeline/data/smarthome_CS_51.json')

# returns JSON object as a dictionary
data = json.load(json_file)

valid_videos = []

# iterates through the json and adding each video id to valid_videos list
for id in data:
    valid_videos.append(id)
    # data_subset = data[id]['subset']
    # print(data_subset)

# lists all files in the RGB directory and saving it into all_videos list    
all_videos = []    
path = '../pipeline/data/RGB/Videos_mp4'
files = os.listdir(path)

for f in files:
    all_videos.append(f[:-4])

# displays length of each video list
print(f"valid vids: {len(valid_videos)}")
print(f"all vids: {len(all_videos)}")

# displays length of matching video list
matching_ids = set(valid_videos) & set(all_videos)
print(f"matching vids: {len(matching_ids)}")

# writes the matching video file paths into a txt file
with open('./sample/valid_videos.txt', 'w') as f:
    for id in matching_ids:
        f.write(f"{path}/{id}.mp4\n")

# closes JSON file
json_file.close()
