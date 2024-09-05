try:
    import cv2
    HAS_CV2 = True 
except:
    HAS_CV2 = False 

import numpy as np
from typing import Tuple

class VideoReader:
    """ Uses cv2 to read video files """
    def __init__(self, filenames: list):
        """ Uses cv2 to open video files and obtain their details for reading

        Parameters
        ------------
        filenames : int
            list of video files
        """
        cumframes = [0]
        containers = []
        Ly = []
        Lx = []
        for f in filenames:  # for each video in the list
            cap = cv2.VideoCapture(f)
            containers.append(cap)
            Lx.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            Ly.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cumframes.append(cumframes[-1] + int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cumframes = np.array(cumframes).astype(int)
        Ly = np.array(Ly)
        Lx = np.array(Lx)
        if (Ly==Ly[0]).sum() < len(Ly) or (Lx==Lx[0]).sum() < len(Lx):
            raise ValueError("videos are not all the same size in y and x")
        else:
            Ly, Lx = Ly[0], Lx[0]

        self.filenames = filenames
        self.cumframes = cumframes 
        self.n_frames = cumframes[-1]
        self.Ly = Ly
        self.Lx = Lx
        self.containers = containers
        self.fs = containers[0].get(cv2.CAP_PROP_FPS)

    def close(self) -> None:
        """
        Closes the video files
        """
        for i in range(len(self.containers)):  # for each video in the list
            cap = self.containers[i]
            cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        The dimensions of the data in the file

        Returns
        -------
        n_frames: int
            The number of frames
        Ly: int
            The height of each frame
        Lx: int
            The width of each frame
        """
        return self.n_frames, self.Ly, self.Lx

    def get_frames(self, cframes):
        """
        read frames "cframes" from videos

        Parameters
        ------------
        cframes : np.array
            start and stop of frames to read, or consecutive list of frames to read
        """
        cframes = np.maximum(0, np.minimum(self.n_frames - 1, cframes))
        cframes = np.arange(cframes[0], cframes[-1] + 1).astype(int)
        # find which video the frames exist in (ivids is length of cframes)
        ivids = (cframes[np.newaxis, :] >= self.cumframes[1:, np.newaxis]).sum(axis=0)
        nk = 0
        im = np.zeros((len(cframes), self.Ly, self.Lx), "uint8")
        for n in np.unique(ivids):  # for each video in cumframes
            cfr = cframes[ivids == n]
            start = cfr[0] - self.cumframes[n]
            end = cfr[-1] - self.cumframes[n] + 1
            nt0 = end - start
            capture = self.containers[n]
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != start:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            fc = 0
            ret = True
            while fc < nt0 and ret:
                ret, frame = capture.read()
                if ret:
                    im[nk + fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    print("img load failed, replacing with prev..")
                    im[nk + fc] = im[nk + fc - 1]
                fc += 1
            nk += nt0
        return im
