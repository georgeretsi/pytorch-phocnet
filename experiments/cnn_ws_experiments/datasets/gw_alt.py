'''
Created on Sep 3, 2017

@author: ssudholt
'''
import os

import numpy as np
from skimage import io as img_io
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

from skimage.transform import resize

from cnn_ws.io.list_io import LineListIO
from cnn_ws.string_embeddings.phoc import build_phoc_descriptor, get_unigrams_from_strings
from cnn_ws.transformations.image_size import check_size
from cnn_ws.transformations.homography_augmentation import HomographyAugmentation


class GWDataset(Dataset):
    '''
    PyTorch dataset class for the segmentation-based George Washington dataset
    '''

    def __init__(self, gw_root_dir, image_extension='.png',
                 cv_split_method=None, cv_split_idx=None,
                 embedding='phoc',
                 phoc_unigram_levels=(1, 2, 4, 8),
                 fixed_image_size=None,
                 min_image_width_height=30):
        '''
        Constructor

        :param gw_root_dir: full path to the GW root dir
        :param image_extension: the extension of image files (default: png)
        :param transform: which transform to use on the images
        :param cv_split_method: the CV method to be used for splitting the dataset
                                if None the entire dataset is used
        :param cv_split_idx: the index of the CV split to be used
        :param partition: the partition of the dataset (train or test)
                          can only be used if cv_split_method and cv_split_idx
                          is not None
        :param min_image_width_height: the minimum height or width a word image has to have
        '''
        # sanity checks

        if embedding not in ['phoc', 'spoc', 'dctow']:
            raise ValueError('embedding must be one of phoc, spoc or dctow')
        if cv_split_method not in [None, 'almazan', 'fifepages']:
            raise ValueError('cv_split_method must be one of None, almazan or fifepages')
        if cv_split_idx is not None and cv_split_method is None:
            raise ValueError('if cv_split_idx is not None, you need to choose a cv_split_method')

        # class members
        self.word_list = None
        self.word_string_embeddings = None
        self.query_list = None
        self.label_encoder = None

        self.fixed_image_size = fixed_image_size

        # load the dataset
        img_filenames = sorted([elem for elem in os.listdir(os.path.join(gw_root_dir, 'pages'))
                                if elem.endswith(image_extension)])
        words = []
        for img_filename in img_filenames:
            page_id = '.'.join(img_filename.split('.')[:-1])
            doc_img = img_io.imread(os.path.join(gw_root_dir, 'pages', img_filename))
            # scale black pixels to 1 and white pixels to 0
            doc_img = 1 - doc_img.astype(np.float32) / 255.0
            annotation_filename = '.'.join(img_filename.split('.')[:-1] + ['gtp'])
            annotation_lines = LineListIO.read_list(os.path.join(gw_root_dir,
                                                                 'ground_truth',
                                                                 annotation_filename))
            # each line is the annotation of a word image in the following format
            #    <ul_x> <ul_y> <lr_x> <lr_y> <transcription>
            for line in annotation_lines:
                ul_x, ul_y, lr_x, lr_y, transcr = line.split(' ')
                ul_x, ul_y, lr_x, lr_y = int(ul_x), int(ul_y), int(lr_x), int(lr_y)
                word_img = doc_img[ul_y:lr_y, ul_x:lr_x].copy()
                word_img = check_size(img=word_img,
                                      min_image_width_height=min_image_width_height)
                #word_img = resize(image=word_img, output_shape=[60, 100]).astype(np.float32)
                words.append((word_img, transcr, page_id))

        self.words = words
        # compute a mapping from class string to class id
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([elem[1] for elem in words])

        # extract unigrams from train split
        unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]
        #unigrams = get_unigrams_from_strings(word_strings=[elem[1] for elem in words])
        # create embedding for the word_list
        self.word_embeddings = None
        word_strings = [elem[1] for elem in words]
        if embedding == 'phoc':
            self.word_embeddings = build_phoc_descriptor(words=word_strings,
                                                                phoc_unigrams=unigrams,
                                                                unigram_levels=phoc_unigram_levels)
        elif embedding == 'spoc':
            raise NotImplementedError()
        else:
            # dctow
            raise NotImplementedError()
        self.word_embeddings = self.word_embeddings.astype(np.float32)

        self.cv_split_method = cv_split_method
        self.cv_split_index = cv_split_idx

        #train_split = None
        #test_split = None
        if cv_split_method is not None:
            if cv_split_method == 'almazan':
                # CV splits as done in Almazan 2014
                self.split_ids = np.load(os.path.join(gw_root_dir, 'almazan_cv_indices.npy'))

            else:
                # fifepages CV
                raise NotImplementedError()

    def mainLoader(self, partition=None, transforms=HomographyAugmentation()):

        self.transforms = transforms
        if partition not in [None, 'train', 'test']:
            raise ValueError('partition must be one of None, train or test')

        if partition is not None:
            if partition == 'train':
                self.word_list = [word for word, split_id in zip(self.words, self.split_ids)
                                  if split_id != self.cv_split_index]
                self.word_string_embeddings = [string for string, split_id in zip(self.word_embeddings, self.split_ids)
                                               if split_id != self.cv_split_index]
            else:
                self.word_list = [word for word, split_id in zip(self.words, self.split_ids)
                                  if split_id == self.cv_split_index]
                self.word_string_embeddings = [string for string, split_id in zip(self.word_embeddings, self.split_ids)
                                               if split_id == self.cv_split_index]
        else:
            # use the entire dataset
            self.word_list = self.words
            self.word_string_embeddings = self.word_embeddings


        if partition == 'test':
            # create queries
            word_strings = [elem[1] for elem in self.word_list]
            unique_word_strings, counts = np.unique(word_strings, return_counts=True)
            qry_word_ids = unique_word_strings[np.where(counts > 1)[0]]

            query_list = np.zeros(len(word_strings), np.int8)
            qry_ids = [i for i in range(len(word_strings)) if word_strings[i] in qry_word_ids]
            query_list[qry_ids] = 1

            self.query_list = query_list
        else:
            word_strings = [elem[1] for elem in self.word_list]
            self.query_list = np.zeros(len(word_strings), np.int8)

        if partition == 'train':
            # weights for sampling
            #train_class_ids = [self.label_encoder.transform([self.word_list[index][1]]) for index in range(len(self.word_list))]
            #word_strings = [elem[1] for elem in self.word_list]
            unique_word_strings, counts = np.unique(word_strings, return_counts=True)
            ref_count_strings = {uword : count for uword, count in zip(unique_word_strings, counts)}
            weights = [1.0/ref_count_strings[word] for word in word_strings]
            self.weights = np.array(weights)/sum(weights)

    def embedding_size(self):
        return len(self.word_string_embeddings[0])


    def __len__(self):
        return len(self.word_list)

    def __getitem__(self, index):
        word_img = self.word_list[index][0]
        if self.transforms is not None:
            word_img = self.transforms(word_img)

        # fixed size image !!!
        word_img = self._image_resize(word_img, self.fixed_image_size)

        word_img = word_img.reshape((1,) + word_img.shape)
        word_img = torch.from_numpy(word_img)
        embedding = self.word_string_embeddings[index]
        embedding = torch.from_numpy(embedding)
        class_id = self.label_encoder.transform([self.word_list[index][1]])
        is_query = self.query_list[index]

        return word_img, embedding, class_id, is_query

    # fixed sized image
    @staticmethod
    def _image_resize(word_img, fixed_img_size):

        if fixed_img_size is not None:
            if len(fixed_img_size) == 1:
                scale = float(fixed_img_size[0]) / float(word_img.shape[0])
                new_shape = (int(scale * word_img.shape[0]), int(scale * word_img.shape[1]))

            if len(fixed_img_size) == 2:
                new_shape = (fixed_img_size[0], fixed_img_size[1])

            word_img = resize(image=word_img, output_shape=new_shape).astype(np.float32)

        return word_img