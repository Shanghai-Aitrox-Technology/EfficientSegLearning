
import os
import json
import numpy as np
from typing import List
import matplotlib.pyplot as plt


from Common.fileio.file_utils import read_txt


def export_feature_distribute(features: np.ndarray, feature_name: str, labels: List,
                              plot_column: int = 3, hist_bins: int = 40):
    """
    Export the mean, std, min, max of features, and plot the histogram distribution.
    :param features: feature array.
    :param feature_name: name of feature.
    :param labels: labels of feature.
    :param plot_column: column of plot figure.
    :param hist_bins: bins of histogram distribution.
    :return: None
    """
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    num_class = len(labels)

    fig_row = int(np.ceil(num_class*1.0/plot_column))
    fig_column = plot_column
    # plt.figure(figsize=(20, 20))
    plt.figure()
    for i in range(num_class):
        plt.subplot(fig_row, fig_column, i+1)
        plt.hist(features[:, i], bins=hist_bins)
        plt.title('{}:{}'.format(feature_name, labels[i]))
    plt.show()

    print('-----------------------------------------------------------------------------------------------------')
    print(f'{feature_name}:')
    max_str = f'{feature_name} max'
    min_str = f'{feature_name} min'
    mean_str = f'{feature_name} mean'
    std_str = f'{feature_name} std'
    for idx, label in enumerate(labels):
        max_str += ' {}:{:.1f}'.format(label, np.max(features[:, idx]))
        min_str += ' {}:{:.1f}'.format(label, np.min(features[:, idx]))
        mean_str += ' {}:{:.1f}'.format(label, np.mean(features[:, idx]))
        std_str += ' {}:{:.1f}'.format(label, np.std(features[:, idx]))

    print(max_str)
    print(min_str)
    print(mean_str)
    print(std_str)


def feature_json2dict(json_dir, txt_path):
    if txt_path is not None:
        file_names = read_txt(txt_path)
    else:
        file_names = os.listdir(json_dir)

    res_dict = {}
    for file_name in file_names:
        if not file_name.endswith('.json'):
            file_name += '.json'
        file_path = json_dir + file_name
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as load_f:
            load_dict = json.load(load_f)
            for key, value in load_dict.items():
                if key not in res_dict:
                    res_dict[key] = [value]
                else:
                    res_dict[key].append(value)

    return res_dict


if __name__ == '__main__':
    FLARE_LABELS_LIST = ['liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava',
                         'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus',
                         'stomach', 'duodenum', 'left kidney']
    json_dir = "/data/zhangfan/FLARE2022/preprocess_data/crop_feature/"
    txt_path = "/data/zhangfan/FLARE2022/file_list/labeled_series.txt"
    res_dict = feature_json2dict(json_dir, txt_path)

    for key, value in res_dict.items():
        if key == 'uid':
            continue
        features = np.array(value)
        if len(features.shape) == 1:
            continue
        if features.shape[1] == 1:
            export_feature_distribute(features, key, ['x'], plot_column=3, hist_bins=40)
        elif features.shape[1] == 3:
            export_feature_distribute(features, key, ['x', 'y', 'z'], plot_column=3, hist_bins=40)
        elif features.shape[1] == 13:
            export_feature_distribute(features, key, FLARE_LABELS_LIST, plot_column=3, hist_bins=40)
