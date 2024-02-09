from pathlib import Path
import os
import shutil

base_dir = Path('/mnt/data/toybox/toybox-5')

for scene_dir in sorted(base_dir.iterdir()):
    img_dir = scene_dir / 'images'
    depth_dir = scene_dir / 'depths'
    segmentation_dir = scene_dir / 'segmentations'

    img_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    segmentation_dir.mkdir(exist_ok=True)

    for file in sorted(scene_dir.iterdir()):
        if os.path.isdir(file):
            continue
        if file.name.startswith('rgba'):
            shutil.move(file, img_dir / file.name)
        elif file.name.startswith('depth'):
            shutil.move(file, depth_dir / file.name)
        elif file.name.startswith('segmentation'):
            shutil.move(file, segmentation_dir / file.name)
    
