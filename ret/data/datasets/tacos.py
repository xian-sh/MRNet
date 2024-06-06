import json
import logging
import torch
from .utils import moment_to_iou2d, bert_embedding, get_vid_feat
from transformers import DistilBertTokenizer
import os

class TACoSDataset(torch.utils.data.Dataset):

    def __init__(self, audio_dir, ann_file, feat_file, num_pre_clips, num_clips):
        super(TACoSDataset, self).__init__()

        self.num_pre_clips = num_pre_clips
        self.num_clips = num_clips
        self.audio_dir = audio_dir
        self.feat_file = feat_file
        with open(ann_file, 'r', encoding='utf-8') as f:
            annos = json.load(f)

        self.annos = annos
        self.data = list(annos.keys())
        logger = logging.getLogger("ret.trainer")

        self.data_new = []
        self.feat_list = []
        self.moments_list = []
        self.all_iou2d_list = []
        self.audios_list = []
        self.num_audios_list = []
        self.sent_list = []
        for vid in self.data:

            duration, timestamps, audios_name, sentences = annos[vid]['num_frames'] / annos[vid]['fps'], annos[vid][
                'timestamps'], annos[vid]['audios'], annos[vid]['sentences']
            feat = get_vid_feat(self.feat_file, vid, self.num_pre_clips, dataset_name="tacos")
            moments = []
            all_iou2d = []
            for timestamp in timestamps:
                time = torch.Tensor([max(timestamp[0]/annos[vid]['fps'], 0), min(timestamp[1]/annos[vid]['fps'], duration)])
                iou2d = moment_to_iou2d(time, self.num_clips, duration)
                moments.append(time)
                all_iou2d.append(iou2d)
            moments = torch.stack(moments)
            all_iou2d = torch.stack(all_iou2d)

            assert moments.size(0) == all_iou2d.size(0)

            num_audios = len(audios_name)
            if num_audios == 1:
                audios = torch.load(os.path.join(self.audio_dir, f'{audios_name[0].split(".")[0]}.pt')).squeeze(
                    dim=1).float()
            elif num_audios > 1:
                audios = [
                    torch.load(os.path.join(self.audio_dir, f'{audio_name.split(".")[0]}.pt')).squeeze(dim=1).float()
                    for audio_name in audios_name]
                audios = torch.squeeze(torch.stack(audios, dim=0), dim=1)
            else:
                raise ValueError("num_audios should be greater than 0!")

            assert moments.size(0) == audios.size(0)
            start_time = moments[:, 0]
            index = torch.argsort(start_time)

            audios = torch.index_select(audios, dim=0, index=index)
            moments = torch.index_select(moments, dim=0, index=index)
            all_iou2d = torch.index_select(all_iou2d, dim=0, index=index)


            # index_1 = index[::4]
            # index_2 = index[1::4]
            # index_3 = index[2::4]
            # index_4 = index[3::4]
            #
            # audios_1 = torch.index_select(audios, dim=0, index=index_1)
            # moments_1 = torch.index_select(moments, dim=0, index=index_1)
            # all_iou2d_1 = torch.index_select(all_iou2d, dim=0, index=index_1)
            #
            # audios_2 = torch.index_select(audios, dim=0, index=index_2)
            # moments_2 = torch.index_select(moments, dim=0, index=index_2)
            # all_iou2d_2 = torch.index_select(all_iou2d, dim=0, index=index_2)
            #
            # audios_3 = torch.index_select(audios, dim=0, index=index_3)
            # moments_3 = torch.index_select(moments, dim=0, index=index_3)
            # all_iou2d_3 = torch.index_select(all_iou2d, dim=0, index=index_3)
            #
            # audios_4 = torch.index_select(audios, dim=0, index=index_4)
            # moments_4 = torch.index_select(moments, dim=0, index=index_4)
            # all_iou2d_4 = torch.index_select(all_iou2d, dim=0, index=index_4)
            #
            #
            # sent_1 = [sentences[i] for i in index_1]
            # sent_2 = [sentences[i] for i in index_2]
            # sent_3 = [sentences[i] for i in index_3]
            # sent_4 = [sentences[i] for i in index_4]
            #
            #
            sent =  [sentences[i] for i in index]

            if 'train' in ann_file:
                for i in range(3, num_audios, 4):
                    self.data_new.append(vid)
                    self.feat_list.append(feat)

                    self.audios_list.append(audios[i-3:i])

                    self.num_audios_list.append(len(audios[i-3:i]))

                    self.moments_list.append(moments[i-3:i])

                    self.all_iou2d_list.append(all_iou2d[i-3:i])

                    self.sent_list.append(sent[i-3:i])
                # self.data_new.append(vid)
                # self.data_new.append(vid)
                # self.data_new.append(vid)
                # self.data_new.append(vid)
                #
                # self.feat_list.append(feat)
                # self.feat_list.append(feat)
                # self.feat_list.append(feat)
                # self.feat_list.append(feat)
                #
                # self.audios_list.append(audios_1)
                # self.audios_list.append(audios_2)
                # self.audios_list.append(audios_3)
                # self.audios_list.append(audios_4)
                #
                #
                # self.num_audios_list.append(len(audios_1))
                # self.num_audios_list.append(len(audios_2))
                # self.num_audios_list.append(len(audios_3))
                # self.num_audios_list.append(len(audios_4))
                #
                # self.moments_list.append(moments_1)
                # self.moments_list.append(moments_2)
                # self.moments_list.append(moments_3)
                # self.moments_list.append(moments_4)
                #
                #
                # self.all_iou2d_list.append(all_iou2d_1)
                # self.all_iou2d_list.append(all_iou2d_2)
                # self.all_iou2d_list.append(all_iou2d_3)
                # self.all_iou2d_list.append(all_iou2d_4)
                #
                # self.sent_list.append(sent_1)
                # self.sent_list.append(sent_2)
                # self.sent_list.append(sent_3)
                # self.sent_list.append(sent_4)

            elif 'test' in ann_file:
                self.data_new.append(vid)
                self.feat_list.append(feat)

                self.audios_list.append(audios)

                self.num_audios_list.append(len(audios))

                self.moments_list.append(moments)

                self.all_iou2d_list.append(all_iou2d)

                self.sent_list.append(sent)



        if 'train' in ann_file:
            self.mode = 'train'
        if 'val' in ann_file:
            self.mode = 'val'
        if 'test' in ann_file:
            self.mode = 'test'

        logger.info("-" * 60)
        logger.info(f"Preparing {len(self.sent_list)} {self.mode} data, please wait...")


    def __getitem__(self, idx):
        vid = self.data_new[idx]


        return self.feat_list[idx], self.audios_list[idx], self.all_iou2d_list[idx], self.moments_list[idx], self.num_audios_list[idx], idx, vid

    def __len__(self):
        return len(self.sent_list)

    def get_duration(self, idx):
        vid = self.data_new[idx]
        return self.annos[vid]['num_frames']/self.annos[vid]['fps']

    def get_sentence(self, idx):

        return self.sent_list[idx]

    def get_moment(self, idx):

        return self.moments_list[idx]

    def get_vid(self, idx):
        vid = self.data_new[idx]
        return vid

    def get_iou2d(self, idx):

        return self.all_iou2d_list[idx]

    def get_num_audios(self, idx):
        return self.num_audios_list[idx]


