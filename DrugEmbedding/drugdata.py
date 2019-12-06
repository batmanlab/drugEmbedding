import torch
from torch.utils.data import Dataset
import os
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np

# reproducibility
#torch.manual_seed(216)
#np.random.seed(216)

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
        self._smiles_class() # split index of FDA drugs, index of ZINC smiles

        self._load_fda_sp()
        self._create_sp_dataset()

        self.sampling_strategy = 'balanced' # simple: positive sample from the nearest neighbor; balanced: positive uniformly sampled from the non-fartherest neighors

    # load drug-drug ATC shortest path
    def _load_fda_sp(self):
        self.df_sp = pd.read_csv(os.path.join(self.fda_drugs_dir, self.fda_drugs_sp_file), index_col=0)

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
            idx = 0
            for l in lines:
                # convert to tokens
                if len(l.split(" ")) == 1: # the SMILES comes from ZINC 250k
                    smi = l # remove /n
                    id = 'zinc_' + str(idx) # use zinc + idx as instance ID
                    idx += 1
                else: # the SMILES comes from FDA drug
                    smi = l.split(" ")[0]
                    id = l.split(" ")[1].lower() # use FDA drug name as instance ID
                    idx += 1
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

    def _smiles_class(self):
        """
        index of FDA drugs, index of ZINC drugs in self.smiles
        :return:
        """
        self.smiles_keys = list(self.smiles.keys())
        self.zinc_idx = [i for i, key in enumerate(self.smiles_keys) if key[:4] == 'zinc']
        self.fda_idx = [i for i, key in enumerate(self.smiles_keys) if key[:4] != 'zinc']

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
        # exclude self-pairs, i.e. sp = 0
        self.df_sp = self.df_sp[self.df_sp['sp']>0]

        # find the nearest neighbors
        min_idx = self.df_sp.groupby('drug_target')['sp'].idxmin()

        df_sp_nn = self.df_sp.loc[min_idx]
        df_sp_fn = self.df_sp.loc[~self.df_sp.index.isin(min_idx)]

        self.fda_atc_drugs = self.df_sp['drug_target'].unique().tolist()
        self.df_sp_nn = df_sp_nn
        self.df_sp_fn = df_sp_fn

        # find the fartherest neighbors
        max_idx = self.df_sp.index[self.df_sp['sp'] == 10]

        # non-fartherest neighbors and assign sampling weights
        self.df_sp_mn = self.df_sp.loc[~self.df_sp.index.isin(max_idx)]
        # for each target drug reweight drug2drug distance by sp, in order to sample sp uniformly
        self.df_sp_mn['sp_count'] = self.df_sp_mn.groupby(['drug_target','sp'])['sp'].transform('count')
        self.df_sp_mn['ttl_count'] = self.df_sp_mn.groupby(['drug_target'])['drug_target'].transform('count')
        self.df_sp_mn['weight'] =self.df_sp_mn['ttl_count'] /  self.df_sp_mn['sp_count']


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

        # ignore idx and sample from weighted dataset
        #if np.random.rand(1) < self.fda_prop: # sample a data point from FDA drugs with probability fda_prop
        #    idx = np.random.choice(self.fda_idx, 1)[0]
        #else:
        #    idx = np.random.choice(self.zinc_idx, 1)[0] # sample a data point from ZINC dataset

        drug_key = self.smiles_keys[idx]

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

        elif self.task in ['atc', 'vae + atc']:
            # if ATC information is available for the sampled drug
            if drug_key in self.fda_atc_drugs:

                loc_ranking_lst = []
                len_lst = []
                loc_sp_lst = []

                if self.sampling_strategy == 'simple':
                    """
                    sampling strategy 1: sample 1 positive example from the target drug's nearest neighbor(s)
                                        & sample nneg negative examples from the target drug's farther neighbor(s)
                    """
                    # sample 1 positive example from the target drug's nearest neighbor(s)
                    pos_sample = self.df_sp_nn[self.df_sp_nn['drug_target'] == drug_key].sample(n=1)
                    pos_key = pos_sample['drug_comparison'].to_numpy()[0]
                    pos_sp = pos_sample['sp'].to_numpy()[0]
                    loc_ranking_lst.append(np.asarray(self.fda_smiles[pos_key]['inputs'])) # always first save positive example
                    len_lst.append(self.fda_smiles[pos_key]['len'])
                    loc_sp_lst.append(pos_sp)

                    # sample nneg negative examples from the target drug's farther neighbor(s)
                    neg_idx = (self.df_sp_fn['drug_target'] == drug_key) & (self.df_sp_fn['drug_comparison'] != pos_key) & (self.df_sp_fn['drug_comparison'] != drug_key) # comparison can not be the positive example and can not be the drug itself
                    neg_sample = self.df_sp_fn[neg_idx].sample(n=self.nneg)
                    neg_keys = neg_sample['drug_comparison']
                    neg_sp = neg_sample['sp']
                    for i, neg_key in enumerate(neg_keys):
                        loc_ranking_lst.append(np.asarray(self.fda_smiles[neg_key]['inputs']))
                        len_lst.append(self.fda_smiles[neg_key]['len'])
                        loc_sp_lst.append(neg_sp.to_numpy()[i])

                elif self.sampling_strategy == 'balanced':
                    """
                    sampling strategy 2: randomly sample 1 positive example from the non-fartherest neighbor(s) (e.g., sp in [2, ,4, 6, 8] but not 10)
                                    & randomly sample nneg examples that are further from the positive example
                    """
                    # sample 1 positive example from non-fartherest neighbor(s)
                    # uniformly sample (sp = 2, 4, 6, 8)
                    pos_sample = self.df_sp_mn[self.df_sp_mn['drug_target'] == drug_key].sample(n=1, weights='weight')
                    #df_sp_mn_target['weight'] = len(df_sp_mn_target.index) / df_sp_mn_target.groupby('sp')['sp'].transform('count')
                    #pos_sample = df_sp_mn_target.sample(n=1, weights='weight')

                    pos_key = pos_sample['drug_comparison'].to_numpy()[0]
                    pos_sp = pos_sample['sp'].to_numpy()[0]
                    loc_ranking_lst.append(np.asarray(self.fda_smiles[pos_key]['inputs'])) # always first save positive example
                    len_lst.append(self.fda_smiles[pos_key]['len'])
                    loc_sp_lst.append(pos_sp)

                    # sample nneg examples that are further from the positive example
                    neg_idx = (self.df_sp['drug_target'] == drug_key) & (self.df_sp['drug_comparison'] != pos_key) & (self.df_sp['drug_comparison'] != drug_key) & (self.df_sp['sp'] > pos_sp)
                    neg_sample = self.df_sp[neg_idx].sample(n=self.nneg)
                    neg_keys = neg_sample['drug_comparison']
                    neg_sp = neg_sample['sp']
                    for i, neg_key in enumerate(neg_keys):
                        loc_ranking_lst.append(np.asarray(self.fda_smiles[neg_key]['inputs']))
                        len_lst.append(self.fda_smiles[neg_key]['len'])
                        loc_sp_lst.append(neg_sp.to_numpy()[i])

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










