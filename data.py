      
import os
import math
import tqdm
import json
import torch
import ffmpeg
import random
import numpy as np
from einops import rearrange
from data_aug import random_aug, data_transform
from torch.utils.data import DataLoader, Dataset


cat2label = {'成品展示': 1, '菜品食用': 2, '食材展示': 3, '后厨制作': 4}
label2cat = {1: '成品展示', 2: '菜品食用', 3: '食材展示', 4: '后厨制作'}

class MyDataset(Dataset):
    def __init__(self, ann_pairs, p=0.1, videos_path='/opt/tiger/debug_server/ByteFood',
                 fps=3, return_vid=False, temporal=False):
        # ann_pairs = [[vid, [[start, end, label], ...]], ...]
        # p: ratio of data augmentation

        self.ann_pairs = ann_pairs
        self.videos_path = videos_path
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)  # batch_size, channel, width, height
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)   # batch_size, channel, width, height
        self.transform = data_transform(p)
        self.p = p
        self.fps = fps
        self.return_vid = return_vid
        self.temporal = temporal

    def __len__(self):
        return len(self.ann_pairs)
    
    def process_video(self, video_path, resize_size=256, crop_size=224, centercrop=True):
        try:
            info = self._get_video_info(video_path)
            h, w = info["height"], info["width"] # 240 426
        except Exception:
            print('ffprobe failed at: {}'.format(video_path))
            return {'video': torch.zeros(1), 'input': video_path,
                    'info': {}}
        if h > w:
            height, width = int(resize_size * (h / w)), resize_size
        else:
            height, width = resize_size, int(resize_size * (w / h))

        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=self.fps)
            .filter('scale', width, height)
        )
        if centercrop:
            x = int((width - crop_size) / 2.0)
            y = int((height - crop_size) / 2.0)
            cmd = cmd.crop(x, y, crop_size, crop_size)
        try:
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
        except ffmpeg.Error as e:
            print("output")
            print(e.stdout)
            print("err")
            print(e.stderr)
        if centercrop and isinstance(crop_size, int):
            height, width = crop_size, crop_size
        video = np.frombuffer(out, np.uint8).reshape(
            [-1, 224, 224, 3])
        video = np.einsum('fhwc->fchw', video)
        return video

    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                                if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(self.convert_to_float(video_stream['avg_frame_rate']))
        try:
            frames_length = int(video_stream['nb_frames'])
            duration = float(video_stream['duration']) # 150.017
        except Exception:
            frames_length, duration = -1, -1
        info = {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}
        return info
    
    def convert_to_float(self, frac_str):
        try:
            return float(frac_str)
        except ValueError:
            try:
                num, denom = frac_str.split('/')
            except ValueError:
                return None
            try:
                leading, num = num.split(' ')
            except ValueError:
                return float(num) / float(denom)
            if float(leading) < 0:
                sign_mult = -1
            else:
                sign_mult = 1
            return float(leading) + sign_mult * (float(num) / float(denom))
    
    def __getitem__(self, index):
        while True:
            video_info = self.ann_pairs[index]
            video_id = video_info[0]
            video_ann = video_info[1]  # [start, end, cat_name]

            x_name = '%s/%s.mp4' % (self.videos_path, video_id)
            if os.path.exists(x_name):
                x = self.process_video(x_name)
                positive_index = list()
                for ann in video_ann:
                    if self.temporal:
                        start = int(ann[0])
                        end = math.ceil(ann[1])
                    else:
                        start = int(ann[0] * self.fps)
                        end = int(ann[1] * self.fps)
                    positive_index.extend(list(range(start, end)))
                break
            else:
                index = (index + 1) % len(self.ann_pairs)

        if random.uniform(0, 1) < self.p:
            x = random_aug(x, self.transform)

        # normalize
        x = x.astype(float) / 255.0
        x = (x - self.mean) / self.std
        
        if self.temporal:
            int_end = x.shape[0] - x.shape[0] % self.fps
            x = x[: int_end]
            # x shape: T, 3(RGB), 224, 224 -> t, 3(RGB), 3(fps), 224, 224
            x = rearrange(x, '(t fps) c h w -> t c fps h w', fps=self.fps)

        positive_index = np.array(positive_index)
        positive_index[positive_index >= x.shape[0] - 1] = x.shape[0] - 1
        label = np.zeros(x.shape[0])
        label[positive_index] = 1
        if self.return_vid:
            return x, label, video_id
        return x, label


def get_train_val_stage_dataset(aug_ratio, stage=1, temporal=False, return_vid=False):
    split_file = './data_ann/dataset_incremental_split.json'
    train_json = './data_ann/douyin_train_4928.json'
    val_json = './data_ann/douyin_val_261.json'
    videos_path = '/opt/tiger/debug_server/ByteFood'

    with open(split_file, 'r') as f:
        train_vids = json.load(f)
    if stage in list(train_vids.keys()):
        train_vids = train_vids[str(stage)]
    else:
        train_vids = sum(list(train_vids.values()), [])
    
    with open(train_json, 'r') as f:
        train_ann = json.load(f)
    
    with open(val_json, 'r') as f:
        val_ann = json.load(f)
    
    train_anns = list()
    for vid in train_vids:
        anns = train_ann[vid]
        refine_anns = list()
        for ann in anns:
            if ann['label'] in list(cat2label.keys()):
                refine_anns.append([ann['start'], ann['end'], ann['label']])
        train_anns.append([vid, refine_anns])

    val_anns = list()
    for vid, anns in val_ann.items():
        refine_anns = list()
        for ann in anns:
            if ann['label'] in list(cat2label.keys()) and cat2label[ann['label']] <= stage:
                refine_anns.append([ann['start'], ann['end'], ann['label']])
        val_anns.append([vid, refine_anns])

    train_set = MyDataset(train_anns, p=aug_ratio, videos_path=videos_path, temporal=temporal, return_vid=return_vid)
    val_set = MyDataset(val_anns, p=0., videos_path=videos_path, temporal=temporal, return_vid=return_vid)
    return train_set, val_set


if __name__ == '__main__':
    train_set, val_set = get_train_val_stage_dataset(aug_ratio=0.1, stage=10, temporal=False, return_vid=True)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6)

    for x, label, vid in tqdm.tqdm(train_loader):
        print(x.shape, label.shape, vid)
    
    for x, label, vid in tqdm.tqdm(val_loader):
        print(x.shape, label.shape, vid)