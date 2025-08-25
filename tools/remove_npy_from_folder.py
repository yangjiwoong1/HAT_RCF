'''
Remove .npy files from AID dataset
'''

import os
import glob

folder_path = 'datasets/AID-dataset/train/LR_x4'

npy_files = glob.glob(os.path.join(folder_path, '*.npy'))

for file in npy_files:  
    try:
        os.remove(file)
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Failed to delete {file}: {e}")

print(f"Total {len(npy_files)} .npy files deleted")
