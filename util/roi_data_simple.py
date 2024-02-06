"""
Simple data structure for ROI data.

Made to emulate ScanImage's RoiDataSimple class.

Author: Flynn OConnell - FlynnOConnell@gmail.com

Notes:
    The image data is stored in a list of lists of numpy arrays.
    The first level is for channels, the second level is for volumes, and the third level is for z-stacks.
    The numpy arrays are the actual image data.
"""


class RoiDataSimple:
    def __init__(self):
        self.hRoi = None
        self.zs = []
        self.channels = []
        self.imageData = [] 

    def add_channel(self):
        self.imageData.append([]) 

    def add_volume_to_channel(self, channel_idx):
        if len(self.imageData) <= channel_idx:
            self.add_channel()
        self.imageData[channel_idx].append([])

    def add_image_to_volume(self, channel_idx, volume_idx, img_data):
        if len(self.imageData) <= channel_idx:
            self.add_channel()
        if len(self.imageData[channel_idx]) <= volume_idx:
            self.add_volume_to_channel(channel_idx)
        self.imageData[channel_idx][volume_idx].append(img_data)

    def cast_image_data(self, new_type):
        """
        Cast the image data to a new type.
        """
        for iterChannels in range(len(self.imageData)):
            for iterVolumes in range(len(self.imageData[iterChannels])):
                for iterZs in range(len(self.imageData[iterChannels][iterVolumes])):
                    self.imageData[iterChannels][iterVolumes][iterZs] = self.imageData[iterChannels][iterVolumes][iterZs].astype(new_type)

    def multiply_image_data(self, factor):
        """
        Multiply the image data by a factor.
        """
        for iterChannels in range(len(self.imageData)):
            for iterVolumes in range(len(self.imageData[iterChannels])):
                for iterZs in range(len(self.imageData[iterChannels][iterVolumes])):
                    self.imageData[iterChannels][iterVolumes][iterZs] *= factor
