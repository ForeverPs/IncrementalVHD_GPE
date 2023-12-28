import os
import json
import tqdm
import ffmpeg
import numpy as np


def get_duration(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams']
                            if stream['codec_type'] == 'video'), None)
    duration = float(video_stream['duration'])
    return duration


def get_video_duration(video_prefix='/opt/tiger/debug_server/ByteFood/'):
    with open('./data_ann/all_highlights_5189.json', 'r') as f:
        jsc = json.load(f)

    vid2duration = dict()
    for k, v in tqdm.tqdm(jsc.items()):
        video_path = '%s%s.mp4' % (video_prefix, k)
        try:
            duration = get_duration(video_path)
            vid2duration[k] = duration
        except:
            print('Invalid Video Path')

    with open('vid2duration.json', 'w') as f:
        json.dump(vid2duration, f)


def incremental_count():
    cat2label = {'成品展示': 1, '菜品食用': 2, '食材展示': 3, '后厨制作': 4}
    with open('./data_ann/douyin_train_4928.json', 'r') as f:
        jsc = json.load(f)
    
    types = list()
    count = 0
    # set 1: [1]
    # set 2: [2], [1,2]
    # set 3: [3], [1, 3], [2, 3], [1, 2, 3]
    # set 4: [4], [2, 4], [3, 4], [1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 3, 4]
    types_count = {i: 0 for i in range(1, 5)}
    dataset = {i: list() for i in range(1, 5)}
    empty = 0
    for vid, anns in jsc.items():
        # if not anns:
        #     empty += 1
        temp_type = list()
        for ann in anns:
            try:
                label = cat2label[ann['label']]
                if label not in temp_type:
                    temp_type.append(label)
            except:
                # print(ann['label'])  # chapter description
                pass
        temp_type = sorted(temp_type)
        if temp_type:
            count += 1
            types_count[max(temp_type)] += 1
            dataset[max(temp_type)].append(vid)
        if temp_type and temp_type not in types:
            types.append(temp_type)
    print(types)
    print(types_count, count)
    print(len(jsc) - count, empty)
    with open('./data_ann/dataset_incremental_split.json', 'w') as f:
        json.dump(dataset, f)
    
    vids = 0
    for k, v in dataset.items():
        vids += len(v)
    print(vids)


def ann_count():
    cat2label = {'成品展示': 1, '菜品食用': 2, '食材展示': 3, '后厨制作': 4}
    with open('./data_ann/all_highlights_5189.json', 'r') as f:
        jsc = json.load(f)
    count = [0, 0, 0, 0]
    for k, v in tqdm.tqdm(jsc.items()):
        for ann in v:
            if ann['label'] in cat2label.keys():
                index = cat2label[ann['label']] - 1
                count[index] += 1
    print(count)


def highlight_duration():
    cat2label = {'成品展示': 1, '菜品食用': 2, '食材展示': 3, '后厨制作': 4}
    with open('./data_ann/all_highlights_5189.json', 'r') as f:
        jsc = json.load(f)
    highlight_durations = list()
    for k, v in tqdm.tqdm(jsc.items()):
        for ann in v:
            if ann['label'] in cat2label.keys():
                start, end = float(ann['start']), float(ann['end'])
                duration = end - start
                if duration:
                    highlight_durations.append(end - start)
    
    highlight_durations = np.array(highlight_durations)
    print(np.min(highlight_durations), np.max(highlight_durations), np.median(highlight_durations))
    np.save('highlights_duration.npy', highlight_durations)


def plot_distribution():
    import matplotlib.pyplot as plt
    
    x = np.load('highlights_duration.npy')
    
    slices = list(range(2, 28, 2))
    ratio, x_ticks, _ = plt.hist(x, bins=slices, edgecolor='#444444', log=True, density=1)
    plt.show()
    plt.close()

    print(len(ratio), len(x_ticks), x_ticks)

    plt.bar(x_ticks[:-1], ratio, width=1.0)
    # plt.xticks(fontsize=12)
    # median_age = np.median(x)
    # plt.axvline(median_age, color='#fc4f30', label='Duration Median', linewidth=2)
    # plt.legend()

    # y_min, y_max = plt.ylim()
    # plt.text(median_age * 1.05, y_max * 0.7, 'Median: {:.1f}'.format(median_age), color='#fc4f30')

    plt.title('Distribution of highlight durations')
    plt.xlabel('duration of highlights')
    plt.ylabel('percentage')
    # plt.tick_params(top='off', right='off')
    # plt.tight_layout()
    plt.savefig('highlights_duration.jpg', dpi=300)
    # plt.show()



if __name__ == '__main__':
    # get_video_statistics()
    # incremental_count()
    # ann_count()
    # highlight_duration()
    # get_video_duration()

    # x = np.load('highlights_duration.npy')
    # print(np.mean(x), len(x))

    # with open('vid2duration.json', 'r') as f:
    #     jsc = json.load(f)
    # durations = list(jsc.values())
    # print(len(durations), np.mean(durations))

    plot_distribution()


