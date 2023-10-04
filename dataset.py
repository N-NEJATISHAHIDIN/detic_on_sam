import numpy as np
import os
import sys
import math
from utils import read_ai2thour_data, draw_images

import torch
from torch.utils.data import Dataset


# import os.path as osp
# from torch_geometric.data import Dataset, download_url


# class MyOwnDataset(Dataset):
#     def __init__(self, root, data_root_path, list_data_ids, transform=None, pre_transform=None, pre_filter=None ):
#         # Initialize your dataset here
#         self.data_root_path = data_root_path
#         self.list_data_ids = list_data_ids
#         super().__init__(root)


#     # @property
#     # def raw_file_names(self):
#     #     return ['some_file_1', 'some_file_2', ...]

#     # @property
#     # def processed_file_names(self):
#     #     return ['data_1.pt', 'data_2.pt', ...]

#     # def process(self):
#     #     idx = 0
#     #     for raw_path in self.raw_paths:
#     #         # Read data from `raw_path`.
#     #         data = Data(...)

#     #         if self.pre_filter is not None and not self.pre_filter(data):
#     #             continue

#     #         if self.pre_transform is not None:
#     #             data = self.pre_transform(data)

#     #         torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
#     #         idx += 1

#     def len(self):
#         return len(self.processed_file_names)

#     def get(self, index):

#         data_id = self.list_data_ids[index][0]
#         instance_id = int(self.list_data_ids[index][1])

#         data_info = read_ai2thour_data(self.data_root_path, data_id)

#         depth_img = data_info['depth'][0]
#         rgb_img = data_info['rgb'][0]
#         seg_img = data_info['seg'][0]
#         point_cloud = data_info['full_xyz_pts'][0]
#         target_obj_id = data_info['data']['descriptions']['target_obj_id'][instance_id]
#         # reference_obj_id = data_info['data']['descriptions']['reference_obj_id'][instance_id]
#         descriptions = data_info['data']['descriptions']['spatial_relation_name'][instance_id].decode("utf-8")
#         target_obj_name = data_info['data']['descriptions']['target_obj_name'][instance_id].decode("utf-8")
#         reference_obj_name = data_info['data']['descriptions']['reference_obj_name'][instance_id].decode("utf-8")
#         # import pdb; pdb.set_trace()

#         # Return a single item from the dataset at the given index
#         return  torch.tensor(depth_img), torch.tensor(rgb_img),torch.tensor(seg_img),torch.tensor(point_cloud), torch.tensor(target_obj_id), descriptions, target_obj_name, reference_obj_name





# Define your custom dataset class
class Ai2Thour_re_dataset(Dataset):
    def __init__(self, data_root_path, list_data_ids):
        # Initialize your dataset here
        self.data_root_path = data_root_path
        self.list_data_ids = list_data_ids

    def __len__(self):
        # Return the length of the dataset
        return len(self.list_data_ids)

    def __getitem__(self, index):

        data_id = self.list_data_ids[index][0]
        instance_id = int(self.list_data_ids[index][1])

        data_info = read_ai2thour_data(self.data_root_path, data_id)

        depth_img = data_info['depth'][0]
        rgb_img = data_info['rgb'][0]
        seg_img = data_info['seg'][0]
        target_obj_id = data_info['data']['descriptions']['target_obj_id'][instance_id]
        saliencies = data_info['saliencies'][instance_id]
        # tsdf_value_pts = data_info['tsdf_value_pts'][0]
        point_cloud = data_info['full_xyz_pts'][0]
        # reference_obj_id = data_info['data']['descriptions']['reference_obj_id'][instance_id]
        descriptions = data_info['data']['descriptions']['spatial_relation_name'][instance_id].decode("utf-8")
        target_obj_name = data_info['data']['descriptions']['target_obj_name'][instance_id].decode("utf-8")
        reference_obj_name = data_info['data']['descriptions']['reference_obj_name'][instance_id].decode("utf-8")

        ################# Visualize ########################
        # import pdb; pdb.set_trace()
        # draw_images([depth_img, rgb_img, seg_img, saliencies, seg_img == target_obj_id], [target_obj_name, descriptions, reference_obj_name], index)

        # Return a single item from the dataset at the given index
        return  torch.tensor(depth_img), torch.tensor(rgb_img),torch.tensor(seg_img),torch.tensor(point_cloud), torch.tensor(target_obj_id), descriptions, target_obj_name, reference_obj_name

