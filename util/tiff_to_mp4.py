import imageio
import cv2
import numpy as np
from pathlib import Path

# Read the .tiff file
path = Path('/data2/fpo/lbm/output/').glob('*.mat*')
tiff_file = list(path)[0]

reader = imageio.get_reader(tiff_file)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4
fps = 30  # frames per second, you can modify this depending on your requirements
height, width = reader.get_data(0).shape[:2]
video_file = path / 'converted.mp4'
out = cv2.VideoWriter(video_file, fourcc, fps, (width, height), isColor=len(reader.get_data(0).shape)==3)

# Iterate through tiff images and write to video
for i, im in enumerate(reader):
    if len(im.shape) == 2:  # grayscale
        im = np.stack([im]*3, axis=2)
    out.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

out.release()
print(f"{tiff_file} has been converted to {video_file}.")