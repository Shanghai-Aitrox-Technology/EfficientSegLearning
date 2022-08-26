
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from Common.fileio.file_utils import read_txt, write_txt


def split_txt(file_path, out_dir, num=4):
    src_name = file_path.split('/')[-1].split('.txt')[0]
    all_names = read_txt(file_path)
    all_num = len(all_names)
    for i in range(num):
        tmp_names = all_names[int(i/num*all_num):int((i+1)/num*all_num)]
        out_path = out_dir + src_name + '_' + str(i) + '.txt'
        write_txt(out_path, tmp_names)


def export_series_list(src_dir, out_path):
    file_names = os.listdir(src_dir)
    file_names = [item.split('.nii.gz')[0] for item in file_names]
    write_txt(out_path, file_names)


if __name__ == '__main__':
    # image_dir = '../raw_data/crop_labeled_image/'
    # out_path = '../raw_data/file_list/labeled_series.txt'
    image_dir = '../raw_data/crop_unlabeled_image/'
    out_path = '../raw_data/file_list/unlabeled_series.txt'
    export_series_list(image_dir, out_path)
