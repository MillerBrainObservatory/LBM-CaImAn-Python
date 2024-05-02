Prepare ROI's
=============

Data Preperation (Pre-Motion Correction)

High Level Overview: 

Steps:
- reshapes the axis to have an x,y,z,t volume 
- sorts the z-planes accoriding to their Y-pixel location 
- calculates and corrects the MROI seams
- calculates and corrects the X-Y shifts across planes 
- outputs data as x-y-t planes or x-y-z-t volumes

Processing Overview
-------------------

The example LBM dataset is a 3D array with dimensions `(25350, 145, 5104)`, where:

- **25350 2D Pages/Frames*: Each page represents a verticle strip of stitched MROI's.
- **145 height**: Vertical dimension of each frame.
- **5104 width**: Horizontal dimension of each frame.

1. **Reshape Based on Planes**

   - **Objective**: To group the data by planes, then by frames within those planes.
   - **New Dimensions**: `(28, 30, 145, 5104)`, calculated by redistributing the 844 frames across the 30 planes, resulting in approximately 28 frames per plane.

2. **Swap Axes for MROI Extraction**

   - **Objective**: To prioritize spatial dimensions (width, height) for slicing based on Y-coordinates.
   - **New Dimensions**: `(28, 5104, 145, 30)`, swapping the planes to the last position.

3. **Channel Ordering**

   - **Objective**: To reorder the planes based on a predefined order (`chans_order`).
   - **Dimensions**: Remain `(28, 5104, 145, 30)` but with planes reordered.

4. **Template Creation (if applicable)**

   - **Objective**: To reduce the dataset to a single frame representing the average across all frames for template creation.
   - **New Dimensions**: `(1, 5104, 145, 30)` after taking the mean across frames.


MROI Handling
-------------

This section focuses on the extraction and processing of multi-regions of interest (MROIs) within the imaging data.

- **Extraction Process**: MROIs are defined by their Y-coordinates and extracted from the volumetric data, adjusting for any flyback lines. This results in arrays with varying sizes in the Y dimension but consistent in other dimensions.
