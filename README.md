# Maskable Retentive Network for Video Moment Retrieval

**Task Example:** The goal of both MR tasks NLMR (_natural language moment retrieval_) and SLMR (_spoken language moment retrieval_) is to predict the temporal boundaries $(\tau_{start}, \tau_{end})$ of target moment described by a given query $q$ (_text or audio modality_).

<p align="center">
 <img src="./assets/task_new1.png" width="70%">
</p>

```
 Two important characteristics:
 1) Temporal association between video clips: The temporal correlation between two video clips that are farther apart is weaker;
 2) Redundant background interference: The background contains a lot of redundant information that can interfere with the recognition of the current event, and this redundancy is even worse in long videos.
```
----------
## Approach

The architecture of the Maskable Retentive Network (MRNet). We conduct modality-specific attention modes, that is, we set _Unlimited Attention_ for language-related attention regions to maximize cross-modal mutual guidance, and perform a new _Maskable Retention_ for video branch $\mathcal{A}(v\to v)$ for enhanced video sequence modeling. 

<div align="center">
  <img src="./assets/main_model.png" alt="Approach" width="800" height="210">
</div>

----------
## To be updated
- [x] : Upload instruction for dataset download
- [x] : Upload implementation
- [x] : Update training and testing instructions
- [x] : Provide access to pre-extracted features of all data
- [ ] : Update trained model


----------

## Download and prepare the datasets

**1. Download the original datasets (optional).** 
   
* The [video feature](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav)  provided by [2D-TAN](https://github.com/microsoft/2D-TAN)
    
        ActivityNet Captions C3D feature
        Charades-STA VGG feature
        TACoS C3D feature

    
* The video I3D feature of Charades-STA dataset from [LGI](https://github.com/JonghwanMun/LGI4temporalgrounding)
     
        wget http://cvlab.postech.ac.kr/research/LGI/charades_data.tar.gz
        tar zxvf charades_data.tar.gz
        mv charades data
        rm charades_data.tar.gz


* The video C3D feature of Charades-STA dataset from [DRN](https://github.com/Alvin-Zeng/DRN)
    
        https://pan.baidu.com/s/1Sn0GYpJmiHa27m9CAN12qw
        password:smil

* The Audio Captions: ActivityNet Speech Dataset: download the [original audio](https://drive.google.com/file/d/11f6sC94Swov_opNfpleTlVGyLJDFS5IW/view?usp=sharing) proposed by [VGCL](https://github.com/marmot-xy/Spoken-Video-Grounding)

* The Audio Captions: Charades-STA Speech Dataset: download the [original audio](https://zenodo.org/record/8019213) proposed by us.

* The Audio Captions: TACoS Speech Dataset: download the [original audio](https://zenodo.org/record/8022063) proposed by us. 

**2. Pre-extracted dataset features.**

        https://pan.baidu.com/xxxx
        password:xxxx
 
**3. Prepare the files in the following structure.**
   
      UniSDNet
      ├── configs
      ├── dataset
      ├── dtfnet
      ├── data
      │   ├── activitynet
      │   │   ├── *text features
      │   │   ├── *audio features
      │   │   └── *video c3d features
      │   ├── charades
      │   │   ├── *text features
      │   │   ├── *audio features
      │   │   ├── *video vgg features
      │   │   ├── *video c3d features
      │   │   └── *video i3d features
      │   └── tacos
      │       ├── *text features
      │       ├── *audio features
      │       └── *video c3d features
      ├── train_net.py
      ├── test_net.py
      └── ···

**4. Or set your own dataset path in the following .py file.**

      dtfnet/config/paths_catalog.py

## Dependencies

    pip install yacs h5py terminaltables tqdm librosa transformers
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    conda config --add channels pytorch
    conda install pytorch-geometric -c rusty1s -c conda-forge


## Training

For training, run the python instruction below:

```
python train_net.py --config-file configs/xxxx.yaml 
```


## Testing
Our trained model are provided in [baiduyun, passcode:d4yl](https://pan.baidu.com/s/1FLzAPACOfcK_xDewZoXAkg?pwd=d4yl) or [Google Drive](xx). Please download them to the `checkpoints/best/` folder.
Use the following commands for testing:

```
python test_net.py --config-file checkpoints/best/xxxx.yaml   --ckpt   checkpoints/best/xxxx.pth
```

## Main NLVG Results:

| **ActivityNet Captions** | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 | mIoU|
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|
| **UniSDNet** | 60.75 | 38.88 | 85.34 | 74.01 | 55.47|
</br>

| **TACoS** | Rank1@0.3 | Rank1@0.5 | Rank5@0.3 | Rank5@0.5 | mIoU|
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|
| **UniSDNet** |  55.56 | 40.26 |  77.08 | 64.01 | 38.88|
</br>

| **Charades-STA (VGG)**  | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 | mIoU|
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|
| **UniSDNet** | 48.41 | 28.33 | 84.76 | 59.46 | 44.41|
</br>

| **Charades-STA (C3D)**  | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 | mIoU|
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|
| **UniSDNet** | 49.57 | 28.39 | 84.70 | 58.49 | 44.29|
</br>

| **Charades-STA (I3D)**  | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 | mIoU|
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|
| **UniSDNet** | 61.02 | 39.70 | 89.97 | 73.20 | 52.69|


## Main SLVG Results:


| **ActivityNet Speech** | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 |Rank5@0.5 | Rank5@0.7 | mIoU|
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **UniSDNet** | 72.27 | 56.29 | 33.29 | 90.41 | 84.28| 72.42 | 52.22|
</br>

| **TACoS Speech** | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 |Rank5@0.5 | Rank5@0.7 | mIoU|
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **UniSDNet** | 51.66 | 37.77 |  20.44 | 76.38 | 63.48 | 33.64 | 36.86 |
</br>

| **Charades-STA Speech(VGG)**  | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 |Rank5@0.5 | Rank5@0.7 | mIoU|
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **UniSDNet** | 60.73 | 46.37 | 26.72 | 92.66 | 82.31 | 57.66 | 42.28 |
</br>

| **Charades-STA (I3D)**  | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 |Rank5@0.5 | Rank5@0.7 | mIoU|
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **UniSDNet** | 67.45 | 53.82 | 34.49 | 94.81 | 87.90 | 69.30 | 48.27 |

## BibTeX 
If you find the repository or the paper useful, please use the following entry for citation.
```
@article{hu2024unified,
  title={Unified Static and Dynamic Network: Efficient Temporal Filtering for Video Grounding},
  author={Jingjing Hu and Dan Guo and Kun Li and Zhan Si and Xun Yang and Xiaojun Chang and Meng Wang},
  year={2024},
  Journal={CoRR},
  volume={abs/2403.14174},
}
```

## Contact
If there are any questions, feel free to contact the author: Jingjing Hu (xianhjj623@gmail.com)

## LICENSE
The annotation files and many parts of the implementations are borrowed from [MMN](https://github.com/MCG-NJU/MMN).
Our codes are under [MIT](https://opensource.org/licenses/MIT) license.

