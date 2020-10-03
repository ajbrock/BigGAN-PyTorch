from torchvision import transforms
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
from PIL import Image
import io


class Kinetics600(Dataset):

    def __init__(self, root, transforms):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.frames = []
        self.labels = []

        with h5py.File(root, 'r') as f:
            self.video_list = f[list(f.keys())[0]]
            for vid_id in list(self.video_list.keys()):
                vid = self.video_list[vid_id]
                for frame in sorted(list(vid.keys())):
                    self.frames.append(vid[frame])
                    self.labels.append(vid_id)
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        frame_name, label = self.frames[index], self.labels[index]
        frame_arr = np.array(self.video_list[label][frame_name])
        frame = self.transforms(Image.open(io.BytesIO(frame_arr)))
        return (frame, label)
