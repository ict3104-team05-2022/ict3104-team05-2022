import glob
import os
import re
import sys

sub_name = re.compile(r"(\w{9})_rgb", flags=re.I).sub
sub_name2 = re.compile(r"(\w{9})_fps", flags=re.I).sub

work_dir = sys.argv[1]

for old_path in glob.glob(os.path.join(work_dir, "*.npy")):
    dirname, old_name = os.path.split(old_path)
    new_name = sub_name("\\1", old_name)
    new_path = os.path.join(dirname, new_name)
    try:
        os.rename(old_path, new_path)
    except OSError as exc:
        print(exc)
    
    # Delete npy file that ends with _fps
    if old_name.endswith("_fps.npy"):
        os.remove(old_path)
    
    if old_name.endswith("_ms.npy"):
        os.remove(old_path)