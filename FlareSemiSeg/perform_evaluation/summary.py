
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.fileio.file_utils import load_df


def get_1fold_average_result(result_path):
    data_df = load_df(result_path)
    labels = ['liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava',
              'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus',
              'stomach', 'duodenum', 'left kidney']
    column_header = []
    for object_name in labels:
        column_header.extend([object_name + '_DSC', object_name + '_NSC'])
    df_column_header = data_df.columns.values
    DSC, NSC = [], []

    for idx, name in enumerate(column_header):
        if name in df_column_header:
            data = data_df[name].values
            valid_data = []
            for i in data:
                if i != -1 and i != float('inf') and not np.isnan(i):
                    valid_data.append(i)
            print('{:}: {:.4f}'.format(name, np.mean(valid_data)))
            if idx % 2 == 0:
                DSC.append(np.mean(valid_data))
            elif idx % 2 != 0:
                NSC.append(np.mean(valid_data))

    print("Mean Dice-{:.4f}, Mean Hausdorff-{:.4f}".format(np.mean(DSC), np.mean(NSC)))
    print("Median Dice-{:.4f}, Median Hausdorff-{:.4f}".format(np.median(DSC), np.median(NSC)))


if __name__ == '__main__':
    result_path = '/data/zhangfan/FLARE2022/results/val_mask/base_flare_val_metric.csv'
    get_1fold_average_result(result_path)