import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.fileio.file_utils import read_csv, write_txt


def select_reliable_series(csv_dir, file_names, out_csv_path, out_txt_path, score_thres=0.9, num_selected=1000):
    if not os.path.exists(out_csv_path):
        all_df = []
        for file_name in file_names:
            file_path = csv_dir + file_name
            df = pd.read_csv(file_path)
            all_df.append(df)
        all_df = pd.concat(all_df, axis=0, ignore_index=True)
        all_df.to_csv(out_csv_path)

    all_df = read_csv(out_csv_path)[1:]
    all_score = {}
    unselected_series_ = []
    for score in all_df:
        if float(score[1]) > score_thres and float(score[2]) == 1:
            all_score[score[0]] = float(score[1])
        else:
            unselected_series_.append(score[0])

    candidates = sorted(all_score.items(), key=lambda item: item[1], reverse=True)
    num_selected = min(num_selected, len(candidates))
    selected_candidates = candidates[:num_selected]
    unselected_candidates = candidates[num_selected:]
    selected_series = [item[0] for item in selected_candidates]
    unselected_series = [item[0] for item in unselected_candidates]
    unselected_series += unselected_series_

    write_txt(out_txt_path+'reliable_series.txt', selected_series)
    write_txt(out_txt_path+'unreliable_series.txt', unselected_series)


if __name__ == "__main__":
    file_dir = '../raw_data/pseudo_mask/iters0_reliable_score/'
    out_csv_path = '../raw_data/pseudo_mask/iters0_reliable_score/reliable_score.csv'
    out_txt_path = '../raw_data/pseudo_mask/iters0_reliable_score/'
    file_names = os.listdir(file_dir)
    select_reliable_series(file_dir, file_names, out_csv_path, out_txt_path, score_thres=0.9, num_selected=1000)