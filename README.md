# Preprocessing code to preprocess 2pRAM (single plane) and LBM recordings

## Light Beads Microscopy (LBM) Analysis Pipeline 



Steps:
- disentangles time and plane dimensions and reshapes to a 4D volume (LBM)
- sorts the z-planes (LBM)
- uses metadata to place MROIs and reconstruct the frames
- calculates and corrects the MROI overlaps and seams
- calculates and corrects the X-Y shifts across planes (LBM)
- outputs data as t-x-y planes or t-x-y-z volumes
- saves a .png with the average frame of all planes
- saves a short .mp4 clip  
