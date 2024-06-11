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

## Download and prepare the datasets

**1. Download the datasets.** 
   
* The [video feature](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav)  provided by [2D-TAN](https://github.com/microsoft/2D-TAN)
    
        ActivityNet Captions C3D feature
        TACoS C3D feature

    
* The video I3D feature of Charades-STA dataset from [LGI](https://github.com/JonghwanMun/LGI4temporalgrounding)
     
        wget http://cvlab.postech.ac.kr/research/LGI/charades_data.tar.gz
        tar zxvf charades_data.tar.gz
        mv charades data
        rm charades_data.tar.gz


* The Audio Captions: ActivityNet Speech Dataset: download the [original audio](https://drive.google.com/file/d/11f6sC94Swov_opNfpleTlVGyLJDFS5IW/view?usp=sharing) proposed by [VGCL](https://github.com/marmot-xy/Spoken-Video-Grounding)

**2. Text and audio feature extraction.** 

```
 cd preprocess
 python text_encode.py
 python audio_encode.py
```

**3. Prepare the files in the following structure.**
   
      UniSDNet
      ├── configs
      ├── dataset
      ├── ret
      ├── data
      │   ├── activitynet
      │   │   ├── *text features
      │   │   ├── *audio features
      │   │   └── *video c3d features
      │   ├── charades
      │   │   ├── *text features
      │   │   └── *video i3d features
      │   └── tacos
      │       ├── *text features
      │       └── *video c3d features
      ├── train_net.py
      ├── test_net.py
      └── ···

**4. Or set your own dataset path in the following .py file.**

      ret/config/paths_catalog.py

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
Our trained model are provided in [Google Drive](xx). Please download them to the `checkpoints/best/` folder.
Use the following commands for testing:

```
python test_net.py --config-file checkpoints/best/xxxx.yaml   --ckpt   checkpoints/best/xxxx.pth
```


## LICENSE
The annotation files and many parts of the implementations are borrowed from [MMN](https://github.com/MCG-NJU/MMN).
Our codes are under [MIT](https://opensource.org/licenses/MIT) license.

