import os
import json
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

class textdata(Dataset):

    def __init__(self, directory, vocab_file, split, max_sequence_length):

        super().__init__()

        #data directory
        self.directory = directory
        self.vocab_file = vocab_file

        #split types
        assert split in ['train','valid','test'], 'Split must be one of: train, valid, test. But received %s'%(split)
        self.split = split

        #max sequence length
        self.max_sequence_length = max_sequence_length

        self._create_text_dataset()


    def _create_text_dataset(self):

        dataset_raw_file = os.path.join(self.directory, 'smiles_' + self.split + '.smi')
        assert os.path.exists(dataset_raw_file), 'File %s not found.'%(dataset_raw_file)

        #initialize data
        self.data = defaultdict(dict)

        #create vocabulary
        self._create_vocab()

        with open(dataset_raw_file, 'r') as file:

            for seq in file.readlines():
                words = list(seq)

                #remove new line char \n
                words = words[:-1]

                #data point id
                id = len(self.data)

                #add sos token
                self.data[id]['words'] = ['<sos>'] + words

                # cut off if exceeds max_sequence_length
                self.data[id]['words'] = self.data[id]['words'][
                                         :self.max_sequence_length - 1]  # -1 to make space for <eos> token

                # add <eos> token
                self.data[id]['words'] += ['<eos>']

                # save length before padding
                self.data[id]['len'] = len(self.data[id]['words']) - 1

                # pad
                self.data[id]['words'].extend(['<pad>'] * (self.max_sequence_length - len(self.data[id]['words'])))

                # convert to idicies
                word_idx = [self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in self.data[id]['words']]

                # create inputs and targets
                self.data[id]['inputs'] = word_idx[:-1]
                self.data[id]['targets'] = word_idx[1:]

                assert self.data[id]['len'] <= self.max_sequence_length


    def _create_vocab(self):

        w2i = dict()
        i2w = dict()

        # add special tokens to vocab
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st]=len(w2i)

        # load unique chars
        char_list = json.load(open(os.path.join(self.directory, self.vocab_file)))
        for i, c in enumerate(char_list):
            i2w[len(w2i)] = c
            w2i[c] = len(w2i)

        self.w2i, self.i2w = w2i, i2w

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_token(self):
        return self.w2i['<pad>']

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def __getitem__(self, idx):

        return {
            'words': self.data[idx]['words'],
            'inputs': np.asarray(self.data[idx]['inputs']),
            'targets': np.asarray(self.data[idx]['targets']),
            'len': self.data[idx]['len']
        }

    def __len__(self):
        return len(self.data)
