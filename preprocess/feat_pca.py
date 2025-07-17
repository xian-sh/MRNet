import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# feat_file = "G:/Dataset/data/TACoS/tall_c3d_features.hdf5"
# target_file = "G:/Dataset/data/TACoS/tall_c3d_features_pca.hdf5"

feat_file = "G:/Dataset/data/Charades_STA/tacos_c3d_fc6_nonoverlap.hdf5"
target_file = "G:/Dataset/data/Charades_STA/tacos_c3d_fc6_nonoverlap_pca.hdf5"

# 创建PCA对象，并指定要降低到的目标维度
n_components = 500
pca = PCA(n_components=n_components)

f = h5py.File(feat_file, 'r')
f2 = h5py.File(target_file, 'a')

feature_all = []
feat_len = []
feat_name = []

for vid in f.keys():
    feat_name.append(vid)
    features = f[vid][:]  # (n, 4096)
    features = np.asarray(features, dtype=np.float32)
    # features = torch.from_numpy(features)
    feature_all.append(features)
    feat_len.append(features.shape[0])


feature_all = np.concatenate(feature_all, axis=0)  # (n, 4096)
print(feature_all.shape)  # tacos torch.Size([88569, 4096])  sta

feature_all = pca.fit_transform(feature_all)  # (n, 500)
print(feature_all.shape)

for i, vid in enumerate(feat_name):
    start = sum(feat_len[:i])
    end = sum(feat_len[:i+1])
    f2.create_dataset(vid, data=feature_all[start:end])




