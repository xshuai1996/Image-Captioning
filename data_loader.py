import torch.utils.data as data
from vocabulary import Vocabulary
from pycocotools.coco import COCO
import nltk
from tqdm import tqdm   # Make your loops show a smart progress meter. See https://tqdm.github.io/
import numpy as np
import json
from PIL import Image
import os
import torch


def get_loader(params, mode):
    """Returns the data loader.
    Args:
      mode: One of 'train' or 'test'.
    """
    if mode == 'train':
        img_folder = params.image_train
        annotations_file = params.annotation_train
        batch_size = params.batch_size_train
        vocab_from_file = False     # NOT load vocabulary from existing file
    if mode == 'test':
        batch_size = params.batch_size_test
        vocab_from_file = True
        img_folder = params.image_test
        annotations_file = params.annotation_test

    # COCO caption dataset.
    dataset = CoCoDataset(params=params,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder,
                          annotations_file=annotations_file)

    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset, batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
            batch_size=dataset.batch_size, drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset, batch_size=dataset.batch_size, shuffle=True,)

    return data_loader


class CoCoDataset(data.Dataset):

    def __init__(self, params, mode, batch_size, vocab_from_file, img_folder, annotations_file):
        self.params = params
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(params, vocab_from_file, annotations_file)
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')

            # Here, nltk.tokenize.word_tokenize(str(sentence).lower()) takes a string type sentence, changes all letters
            # into lower case and returns a list containing all words in that sentence.
            # Use np.arange because it runs faster than range
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index
                          in tqdm(np.arange(len(self.ids)))]

            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())

            # paths stores names of all images in test set (e.g. COCO_test2014_000000264794.jpg)
            self.paths = [item['file_name'] for item in test_info['images']]


    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.params.transform_train(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.params.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.params.end_word))
            # transfer to long tensor because class Embedding takes long tensor as input
            caption = torch.Tensor(caption).long().to(self.params.device)
            return image, caption

        # obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.params.transform_test(PIL_image)
            return orig_image, image


    def get_train_indices(self):
        # the lengths of captions in training set vary greatly from around 5 to around 55.
        # in every training step we want to randomly choose a length, then filter out all captions have this length,
        # then randomly select the number of captions among them.
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices


    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)