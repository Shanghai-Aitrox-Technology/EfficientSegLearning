# FLARE Semi-Supervised Segmentation

## 1. Data Prepare
### Download FLARE22 dataset
Registory and download [FLARE22](https://flare22.grand-challenge.org/) dataset

### Data folder structure
- Raw data directory: './FlareSemiSeg/raw_data'
- Put raw data into folder as follows:
```wiki
├── labeled_image (total 50)
   ├── FLARE22_Tr_0001_0000.nii.gz
├── labeled_mask (total 50)
   ├── FLARE22_Tr_0001.nii.gz
├── unlabeled_image (total 2000)
   ├── Case_00001_0000.nii.gz
├── test_image (total 50)
   ├──FLARETs_0001_0000.nii.gz
```

### Data preprocess
- Reorients image to standard radiology view.
- Remove background by threshold segmentation.
- Perform the preprocess in './FlareSemiSeg/preprocess' folder:
```
python 6_threshold_seg.py
```
- Extract file names list in './FlareSemiSeg/preprocess' folder:
```
python 4_txt_tools.py
```
- The generated dataset is as follows:
```wiki
├── crop_labeled_image (total 50)
   ├── FLARE22_Tr_0001.nii.gz
├── crop_labeled_mask (total 50)
   ├── FLARE22_Tr_0001.nii.gz
├── crop_unlabeled_image (total 2000)
   ├── Case_00001.nii.gz
├── file_list
   ├── labeled_series.txt
   ├── unlabeled_series.txt
```

### Generate the labeled training data
- Perform the data generation in './FlareSemiSeg/data_prepare' folder:
Edit the 'full_config.yaml' file and set {config_path} in the 'run.py' file.
```
python run.py
```
- The generated dataset is as follows:
```wiki
├── gen_data/full_crop_train
    ├── coarse_image
    ├── coarse_mask
    ├── fine_image
    ├── fine_mask
    ├── file_list
        ├── train_series_uid.txt
        ├── val_series_uid.txt
    ├── db
        ├── seg_train_fold_1
        ├── seg_val_fold_1
```

## 2. Supervised training
### Training coarse segmentation model
- Perform the coarse model training in './FlareSemiSeg/coarse_seg' folder:
Edit the 'full_config.yaml' file and set {config_path} in the 'run.py' file.
```
bash run.sh
```
- Log (Default: flare_full_coarse_seg_result) is saved in the current 'output' folder.
### Training fine segmentation model
- Perform the fine model training in './FlareSemiSeg/fine_seg' folder:
Edit the 'full_config.yaml' file and set {config_path} in the 'run.py' file.
```
bash run.sh
```
- Log (Default: flare_full_fine_seg_result) is saved in the current 'output' folder.
- The items in log file is as follows:
```wiki
├── codes (copy for reproduction)
├── logs (tensorboard visulation)
├── models (checkpoints: best_model.pt, epoch_360_model.pt,epoch_660_model.pt, epoch_1000_model.pt)
```
- The checkpoint of best_model is used for generating pseudo image on unlabeled images, and epochs of 360, 660, 1000 are used to generate the reliable score.
### Select the reliable pseudo cases
- Perform the inference in 'FlareSemiSeg/select_reliable_infer' folder.
- Edit the following items in 'config.yaml' file.
```wiki
coarse_model.weight_path (path of coarse best model)
fine_model.weight_path
fine_model.weight_names(best_model.pt, epoch_360_model.pt, epoch_660_model.pt, epoch_1000_model.pt)
testing.save_dir
testing.save_reliable_path
```
- Generate the pseudo image and reliable score.
```
python run.py
```
- Select the 50% of most reliable cases.
```
python select_reliable_series.py
```
- Selected series is saved in 'reliable_series.txt' file.


## 2. Semi-Supervised training
### Generate the pseudo training dataset
- Perform the data generation in "./FlareSemiSeg/data_prepare" folder.
- Edit the 'pseudo_config.yaml' file and set {config_path} in the 'run.py' file.
```
python run.py
```
- Results are saved in './gen_data/pseudo_crop_train_iters0' folder.

### Training coarse semi-segmentation model
- Perform the coarse model training in './FlareSemiSeg/coarse_seg' folder:
Edit the 'semi_config.yaml' file and set {config_path} in the 'run.py' file.
```
bash run.sh
```
- Log (Default: flare_semi_coarse_seg_iters0) is saved in the current 'output' folder.
### Training fine semi-segmentation model
- Perform the fine model training in './FlareSemiSeg/fine_seg' folder:
Edit the 'semi_config.yaml' file and set {config_path} in the 'run.py' file.
```
bash run.sh
```
- Log (Default: flare_semi_fine_seg_iters0) is saved in the current 'output' folder.
- The total epoch of semi-supervised is 100, so the output of checkpoints is  best_model.pt, epoch_36_model.pt,epoch_66_model.pt, epoch_100_model.pt
### Select the reliable pseudo cases
- Perform the inference in 'FlareSemiSeg/select_reliable_infer' folder.
- Edit the following items in 'config.yaml' file.
```wiki
coarse_model.weight_path (path of coarse best model)
fine_model.weight_path
fine_model.weight_names(best_model.pt, epoch_36_model.pt, epoch_66_model.pt, epoch_100_model.pt)
testing.save_dir
testing.save_reliable_path
```
- Generate the pseudo image and reliable score.
```
python run.py
```
- Select the cases of whcih reliable score exceed 0.9.
```
python select_reliable_series.py
```
- Selected series is saved in 'reliable_series.txt' file.

## 3. Iterative Semi-supervised training
- Repeat or Loop the above setp 2 for iterative training until the metric isn't improved or the total iterative times is reached.
- Empirically, the 4~5 times of iterative training could reach the satisfying result.