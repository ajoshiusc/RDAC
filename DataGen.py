## Deep Active Lesion Segmention (DALS), Code by Ali Hatamizadeh ( http://web.cs.ucla.edu/~ahatamiz/ )

from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy as np
import h5py

class BaseDataProvider(object):

    channels = 1
    n_class = 2
    def _load_data_and_label(self):
        train_data, labels,shape = self._next_data()
        nx = train_data.shape[1]
        ny = train_data.shape[0]
        train_data = train_data.reshape(ny, nx, self.channels)
        labels = labels.reshape(ny, nx, self.n_class)

        return train_data, labels,shape
    def __call__(self, n):
        train_data, labels,shape = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[0]
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels,shape  = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels

        return X, Y,shape

class ImageGenFromFiles(BaseDataProvider):

    def __init__(self, search_path,data_suffix, mask_suffix,
                 shuffle_data, n_class):
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.data_files = self._find_data_files(search_path)
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        assert len(self.data_files) > 0
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def _find_data_files(self, search_path):
        all_files= [os.path.join(path, file) for (path, dirs, files) in os.walk(search_path)for file in files]

        return all_files #[name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]

    def _load_file(self, path):
        #image = np.load(path)
        h5 = h5py.File(path, 'r')
        image = h5.get('X')
        image = np.array(image)


        image = image.astype('float32')
        image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))

        return image

    def _load_label(self, path):
        #label = np.load(path)
        h5 = h5py.File(path, 'r')
        label = h5.get('Y')
        label = np.array(label)

        label = label.astype('float32')
        label *= 1.0 / label.max()

        return label,label.shape

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
    def _next_data(self):

        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        img = self._load_file(image_name)
        label,shape = self._load_label(label_name)

        return img, label,shape



class ImageGen(BaseDataProvider):

    def __init__(self, file_name, shuffle_data, n_class):
        self.data_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        #data = self._load_file(file_name)
        h5 = h5py.File(file_name, 'r')

        self.images = np.float32(np.array(h5.get('X')[:,:,:,0]))/256.0
        self.labels = np.float32(np.array(h5.get('Y')))/256.0

        self.data_ids = np.array(range(self.images.shape[0]))
        if self.shuffle_data:
            np.random.shuffle(self.data_ids)

        assert len(self.data_ids) > 0


        self.channels = 1 if len(self.images.shape) == 3 else self.images.shape[-1]

    '''def _find_data_files(self, search_path):
        all_files= [os.path.join(path, file) for (path, dirs, files) in os.walk(search_path)for file in files]

        return all_files #[name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]

    def _load_file(self, path):
        #image = np.load(path)
        h5 = h5py.File(path, 'r')
        image = h5.get('X')
        image = np.array(image)


        image = image.astype('float32')
        image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))

        return image

    def _load_label(self, path):
        #label = np.load(path)
        h5 = h5py.File(path, 'r')
        label = h5.get('Y')
        label = np.array(label)

        label = label.astype('float32')
        label *= 1.0 / label.max()

        return label,label.shape'''

    def _cylce_data(self):
        self.data_idx += 1
        if self.data_idx >= self.images.shape[0]:
            self.data_idx = 0
            if self.shuffle_data:
                np.random.shuffle(self.data_ids)


    def _next_data(self):

        self._cylce_data()
        img = self.images[self.data_ids[self.data_idx]]
        label = self.labels[self.data_ids[self.data_idx]] 

        shape = label.shape[0]

        return img, label,shape


