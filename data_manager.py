# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path

import numpy as np

import h5py
import numpy as np

import torch
from torch.utils import data


class HDF5SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super().__init__()
        self.file = h5py.File(in_file, 'r')['symworld/']
        self.episode = None

    def __getitem__(self, index):
        ret = {}

        obs = self.states[index, ...].astype(np.float32)
        obs = self.normalize_image(obs)

        ret['obs_set'] = np.moveaxis(obs, -1, 0)  # move from (x, y, c) to (c, x, y)
        ret['reward_set'] = (self.rewards[index] / 100).astype(np.float32)  # Reward between [-1, 1]
        ret['action_set'] = self.actions[index].astype(np.float32)

        # Observation at next time step
        obs_t1 = self.states[index + 1, ...].astype(np.float32)
        obs_t1 = self.normalize_image(obs_t1)
        ret['obs_set_t1'] = np.moveaxis(obs_t1, -1, 0)  # move from (x, y, c) to (c, x, y)

        return ret

    def __len__(self):
        return self.ep_length

    def set_episode(self, ep):
        self.episode = self.file['ep{}'.format(ep)]
        self.rewards = self.episode['reward_set']
        self.actions = self.episode['action_set']
        self.states = self.episode['obs_set']

        # should be c_nonzero + 1, but since last state is not included, cancels out
        reward_set = self.episode['reward_set']
        self.key_idx = np.where(reward_set != 0)
        self.ep_length = self.actions.shape[0]

    def normalize_image(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).

        https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    """

    def __init__(self, file_path, load_data=False, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())

        files = sorted(p.glob('*.hdf5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di[
                                                                                                                 'file_path'] ==
                                                                                                             removal_keys[
                                                                                                                 0] else di
                for di
                in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


''' OLD Tensorflow dataloader, no longer needed '''


class DataManager(object):
    def load(self):
        # Load dataset
        dataset_zip = np.load('data/megaman.npz')
        self.imgs = dataset_zip['arr_0']

        # dataset_zip = h5py.File('data/megaman.hdf5', 'r')
        # self.imgs       = dataset_zip['megaman']
        self.n_samples = self.imgs.shape[0]
        # 27312 for megaman

    @property
    def sample_size(self):
        return self.n_samples

    def get_image(self, index):
        img = self.imgs[index]
        image_size = self.input_width * self.input_height * self.input_channels
        img = img.reshape(image_size)
        return img

    def get_images(self, indices):
        images = []
        for index in indices:
            img = self.get_image(index)
            images.append(img)
        return images

    def get_random_images(self, size):
        indices = [np.random.randint(self.n_samples) for i in range(size)]
        return self.get_images(indices)
