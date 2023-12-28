      
import os
import math
import tqdm
import json
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader, Dataset


cat2label = {'成品展示': 1, '菜品食用': 2, '食材展示': 3, '后厨制作': 4}
label2cat = {1: '成品展示', 2: '菜品食用', 3: '食材展示', 4: '后厨制作'}

class MyDataset(Dataset):
    def __init__(self, ann_pairs, videos_path='/opt/tiger/debug_server/ByteFood_feat',
                 fps=3, return_vid=False, temporal=False):
        # ann_pairs = [[vid, [[start, end, label], ...]], ...]
        # p: ratio of data augmentation

        self.ann_pairs = ann_pairs
        self.videos_path = videos_path
        self.fps = fps
        self.return_vid = return_vid
        self.temporal = temporal

    def __len__(self):
        return len(self.ann_pairs)
    
    def __getitem__(self, index):
        while True:
            video_info = self.ann_pairs[index]
            video_id = video_info[0]
            video_ann = video_info[1]  # [start, end, cat_name]

            x_name = '%s/%s.npy' % (self.videos_path, video_id)
            if os.path.exists(x_name):
                x = np.load(x_name)
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
        
        if self.temporal:
            int_end = x.shape[0] - x.shape[0] % self.fps
            x = x[: int_end]
            x = rearrange(x, '(t fps) c -> t c fps', fps=self.fps)

        positive_index = np.array(positive_index)
        positive_index[positive_index >= x.shape[0] - 1] = x.shape[0] - 1
        label = np.zeros(x.shape[0])
        label[positive_index] = 1
        if self.return_vid:
            return x, label, video_id
        return x, label


def get_train_val_stage_dataset(stage=1, temporal=False, return_vid=False, including=False):
    split_file = './data_ann/dataset_incremental_split.json'
    train_json = './data_ann/douyin_train_4928.json'
    val_json = './data_ann/douyin_val_261.json'
    videos_path = '/opt/tiger/debug_server/ByteFood_feat'

    with open(split_file, 'r') as f:
        train_vids = json.load(f)

    if including:
        my_train_vids = list()
        for i in range(1, stage + 1):
            if str(i) in list(train_vids.keys()):
                my_train_vids.extend(train_vids[str(i)])
        train_vids = my_train_vids.copy()
    else:
        if str(stage) in list(train_vids.keys()):
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

    train_set = MyDataset(train_anns, videos_path=videos_path, temporal=temporal, return_vid=return_vid)
    val_set = MyDataset(val_anns, videos_path=videos_path, temporal=temporal, return_vid=return_vid)
    return train_set, val_set


if __name__ == '__main__':
    train_set, val_set = get_train_val_stage_dataset(stage=10, temporal=False, return_vid=True)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6)

    for x, label, vid in tqdm.tqdm(train_loader):
        print(x.shape, label.shape, vid)
    
    for x, label, vid in tqdm.tqdm(val_loader):
        print(x.shape, label.shape, vid)