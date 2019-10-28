from torch.utils.data import Dataset
import os
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np

class drugdata(Dataset):

    def __init__(self, task, fda_drugs_dir, fda_smiles_file, fda_vocab_file, fda_drugs_sp_file,
                 experiment_dir, smi_file,
                 max_sequence_length, nneg=None):
        super(drugdata, self).__init__()

        self.task = task
        self.fda_drugs_dir = fda_drugs_dir
        self.fda_smiles_file = fda_smiles_file
        self.fda_vocab_file = fda_vocab_file
        self.fda_drugs_sp_file = fda_drugs_sp_file

        self.experiment_dir = experiment_dir
        self.smi_file = smi_file

        self.max_sequence_length = max_sequence_length
        self.nneg = nneg

        self._create_vocab()
        self._create_fda_smiles_dataset() # create FDA drugs SMILES only dataset
        self._create_smiles_dataset() # create train, valid, test SMILES dataset

        self._load_fda_sp()
        self._create_sp_dataset()

    # load drug-drug ATC shortest path
    def _load_fda_sp(self):
        with open(os.path.join(self.fda_drugs_dir, self.fda_drugs_sp_file), 'rb') as fp:
            self.fda_sp_dict = pickle.load(fp)

    # load FDA drug smiles
    def _load_fda_smiles(self):
        fda_smiles_dict = {}
        with open(os.path.join(self.experiment_dir, self.smi_file), 'r') as fp:
            idx = 1
            for line in fp.readlines():
                if line != '\n':
                    words = line.split(" ")
                    if len(words) == 1: # the SMILES comes from ZINC 250K
                        fda_smiles_dict['zinc_' + str(idx)] = line[:-1]
                        idx += 1
                    else: # the SMILES comes from FDA drugs
                        fda_smiles_dict[words[1]] = words[0]
        fp.close()
        self.fda_smiles_dict = fda_smiles_dict

    def _create_vocab(self):

        w2i = dict()
        i2w = dict()

        # add special tokens to vocab
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st]=len(w2i)

        # load unique chars
        with open(os.path.join(self.fda_drugs_dir, self.fda_vocab_file), 'rb') as fp:
            char_list = pickle.load(fp)
        fp.close()

        for i, c in enumerate(char_list):
            i2w[len(w2i)] = c
            w2i[c] = len(w2i)

        self.w2i, self.i2w = w2i, i2w

    def _smiles_to_tokens(self, s):
        s = s.strip()

        j = 0
        tokens_lst = []
        while j < len(s):
            # handle atoms with two characters
            if j < len(s) - 1 and s[j:j + 2] in self.w2i.keys():
                token = s[j:j + 2]
                j = j + 2

            # handle atoms with one character including hydrogen
            elif s[j] in self.w2i.keys():
                token = s[j]
                j = j + 1

            # handle unknown characters
            else:
                token = '<unk>'
                j = j + 1

            tokens_lst.append(token)

        return tokens_lst

    def _create_smiles_dataset(self):
        """
        create SMILES dataset:
        key: drug name
        inputs: from <sos> idx to the idx of max length
        targets: from the first atom idx to the idx of max length
        len: length from <sos> to the last atom token in the SMILES
        :return:
        """

        self.smiles = defaultdict(dict)

        with open(os.path.join(self.experiment_dir, self.smi_file)) as file:

            lines = file.read().splitlines()
            idx = 1
            for l in lines:
                # convert to tokens
                if len(l.split(" ")) == 1: # the SMILES comes from ZINC 250k
                    smi = l # remove /n
                    id = 'zinc_' + str(idx) # use zinc + idx as instance ID
                    idx += 1
                else: # the SMILES comes from FDA drug
                    smi = l.split(" ")[0]
                    id = l.split(" ")[1].lower() # use FDA drug name as instance ID
                words = self._smiles_to_tokens(smi)

                #add sos token
                self.smiles[id]['words'] = ['<sos>'] + words

                # cut off if exceeds max_sequence_length
                self.smiles[id]['words'] = self.smiles[id]['words'][
                                         :self.max_sequence_length - 1]  # -1 to make space for <eos> token

                # add <eos> token
                self.smiles[id]['words'] += ['<eos>']

                # save length before padding
                self.smiles[id]['len'] = len(self.smiles[id]['words']) - 1

                # pad
                self.smiles[id]['words'].extend(['<pad>'] * (self.max_sequence_length - len(self.smiles[id]['words'])))

                # convert to idicies
                word_idx = [self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in self.smiles[id]['words']]

                # create inputs and targets
                self.smiles[id]['inputs'] = word_idx[:-1]
                self.smiles[id]['targets'] = word_idx[1:]

                assert self.smiles[id]['len'] <= self.max_sequence_length
        file.close()

    def _create_fda_smiles_dataset(self):
        self.fda_smiles = defaultdict(dict)

        with open(os.path.join(self.fda_drugs_dir, self.fda_smiles_file)) as file:

            lines = file.read().splitlines()
            for l in lines:
                # convert to tokens
                smi = l.split(" ")[0]
                id = l.split(" ")[1].lower() # use FDA drug name as instance ID
                words = self._smiles_to_tokens(smi)

                #add sos token
                self.fda_smiles[id]['words'] = ['<sos>'] + words

                # cut off if exceeds max_sequence_length
                self.fda_smiles[id]['words'] = self.fda_smiles[id]['words'][
                                         :self.max_sequence_length - 1]  # -1 to make space for <eos> token

                # add <eos> token
                self.fda_smiles[id]['words'] += ['<eos>']

                # save length before padding
                self.fda_smiles[id]['len'] = len(self.fda_smiles[id]['words']) - 1

                # pad
                self.fda_smiles[id]['words'].extend(['<pad>'] * (self.max_sequence_length - len(self.fda_smiles[id]['words'])))

                # convert to idicies
                word_idx = [self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in self.fda_smiles[id]['words']]

                # create inputs and targets
                self.fda_smiles[id]['inputs'] = word_idx[:-1]
                self.fda_smiles[id]['targets'] = word_idx[1:]

                assert self.fda_smiles[id]['len'] <= self.max_sequence_length
        file.close()

    def _create_sp_dataset(self):
        """
        prepare dataset for local ranking decision
        :return:
        """
        # convert dictionary to pandas dataframe
        df_sp = pd.DataFrame(list(self.fda_sp_dict.items()), columns=['drugs', 'sp'])

        drugs = list(df_sp['drugs'])
        drug_target = [d[0] for d in drugs]
        drug_comparison = [d[1] for d in drugs]

        df_sp['drug_target'] =drug_target
        df_sp['drug_comparison'] = drug_comparison

        # exclude self-pairs, i.e. sp = 0
        df_sp = df_sp[df_sp['sp']>0]

        # find the nearest neighbors
        min_sp = df_sp.groupby('drug_target')['sp'].min()
        drugs_min_sp = pd.DataFrame({'drug_target':min_sp.index, 'sp':min_sp.values})
        df_sp_nn = df_sp.merge(drugs_min_sp, how='inner', on=('drug_target', 'sp')) # nearest neighbors

        # find the farther neighbors
        df_sp_fn = df_sp[(~df_sp.drugs.isin(df_sp_nn.drugs))]

        self.fda_atc_drugs = df_sp['drug_target'].unique().tolist()
        self.df_sp = df_sp
        self.df_sp_nn = df_sp_nn
        self.df_sp_fn = df_sp_fn



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

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        """
        for each target drug:
        sample 1 drug from the target drug's nearest neighbor(s) in ATC hierarchy, e.g. drugs under the same parent node
        sample n drugs from other branches of the ATC hierarchy
        :param idx:
        :return:
        """

        keys = list(self.smiles.keys())
        drug_key = keys[idx]

        drug_smiles = self.smiles[drug_key]['words']
        drug_inputs = np.asarray(self.smiles[drug_key]['inputs'])
        drug_targets = np.asarray(self.smiles[drug_key]['targets'])
        drug_len = self.smiles[drug_key]['len']

        drug_dict = {}
        if self.task == 'vae': # if the task is VAE only, i.e. recon. loss & KL loss
            drug_dict['drug_name'] = drug_key
            drug_dict['drug_smiles'] = drug_smiles
            drug_dict['drug_inputs'] = drug_inputs
            drug_dict['drug_targets'] = drug_targets
            drug_dict['drug_len'] = drug_len

        elif self.task == 'vae + atc':
            # if ATC information is available for the sampled drug
            if drug_key in self.fda_atc_drugs:
                # sample 1 positive examples from the target drug's nearest neighbor(s)
                loc_ranking_lst = []
                len_lst = []
                loc_sp_lst = []

                pos_sample = self.df_sp_nn[self.df_sp_nn['drug_target'] == drug_key].sample(n=1)
                pos_key = pos_sample['drug_comparison'].get_values()[0]
                pos_sp = pos_sample['sp'].get_values()[0]
                loc_ranking_lst.append(np.asarray(self.fda_smiles[pos_key]['inputs'])) # always first save positive example
                len_lst.append(self.fda_smiles[pos_key]['len'])
                loc_sp_lst.append(pos_sp)


                # sample nneg negative examples from the target drug's farther neighbor(s)
                neg_sample = self.df_sp_fn[self.df_sp_fn['drug_target'] == drug_key].sample(n=self.nneg)
                neg_keys = neg_sample['drug_comparison']
                neg_sp = neg_sample['sp']
                for i, neg_key in enumerate(neg_keys):
                    loc_ranking_lst.append(np.asarray(self.fda_smiles[neg_key]['inputs']))
                    len_lst.append(self.fda_smiles[neg_key]['len'])
                    loc_sp_lst.append(neg_sp.get_values()[i])

                loc_ranking_indicator = np.ones(1, dtype=np.long) # indicator that local ranking info is available
                loc_ranking_inputs = np.stack(loc_ranking_lst)
                loc_ranking_len = np.array(len_lst)
                loc_ranking_sp = np.array(loc_sp_lst)

            # if ATC information is not available for the sample drug
            else:
                loc_ranking_indicator = np.zeros(1, dtype=np.long) # indicator that local ranking info not available
                loc_ranking_inputs = np.zeros((1+self.nneg, self.max_sequence_length-1), dtype=np.long)
                loc_ranking_len = np.zeros(1+self.nneg, dtype=np.long)
                loc_ranking_sp = np.zeros(1+self.nneg, dtype=np.long)

            drug_dict['drug_name'] = drug_key
            drug_dict['drug_smiles'] = drug_smiles
            drug_dict['drug_inputs'] = drug_inputs
            drug_dict['drug_targets'] = drug_targets
            drug_dict['drug_len'] = drug_len
            drug_dict['loc_ranking_indicator'] = loc_ranking_indicator
            drug_dict['loc_ranking_inputs'] = loc_ranking_inputs
            drug_dict['loc_ranking_len'] = loc_ranking_len
            drug_dict['loc_ranking_sp'] = loc_ranking_sp

        return drug_dict










