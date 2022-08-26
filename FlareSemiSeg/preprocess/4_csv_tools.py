
import os
import pandas as pd


def merge_csv(csv_dir, file_names, out_path):
    all_df = []
    for file_name in file_names:
        file_path = csv_dir + file_name
        df = pd.read_csv(file_path)
        all_df.append(df)
    all_df = pd.concat(all_df, axis=0, ignore_index=True)
    all_df.to_csv(out_path)


if __name__ == "__main__":
    file_dir = '/data/zhangfan/FLARE2022/file_list/crop_unlabeled_reliable_score_iters1/'
    out_path = '/data/zhangfan/FLARE2022/file_list/crop_unlabeled_reliable_score_iters1/reliable_score.csv'
    file_names = os.listdir(file_dir)
    merge_csv(file_dir, file_names, out_path)