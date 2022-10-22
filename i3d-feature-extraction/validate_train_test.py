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

# closes JSON file
json_file.close()

# lists all files in the RGB directory and saving it into extracted_videos list
extracted_videos = []
# path = './output/RGB'
path = '../pipeline/data/v_iashin_i3d'
files = os.listdir(path)
print(files)

for f in files:
    extracted_videos.append(f.split('.')[0])

print(f'Extracted Video: {extracted_videos}')
# displays length of each video list
print(f"current vids: {len(valid_videos)}")
print(f"extracted vids: {len(extracted_videos)}")

extracted_set = set(extracted_videos)
missing_vids = []

# removes video IDs that do not have RGB npy files
for id in valid_videos:
    if id not in extracted_videos:
        data.pop(id)
        missing_vids.append(id)
        
# creates new updated version of smarthome.json with removed video IDs
with open('../pipeline/data/smarthome_CS_51_v2.json', "w") as outfile:
    json.dump(data, outfile)

print(f"missing vids: {', '.join(missing_vids)}")
print(f"\n{len(missing_vids)} vids removed from smarthome_CS_51_v2.json")
print(f"smarthome_CS_51_v2.json saved to ./pipeline/data/")