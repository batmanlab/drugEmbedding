from utils import *
from evae import *
from hvae import *
from drugdata import *
from metrics import fda_drug_rep, dendrogram_purity_score
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

import os
import time
import json
import random
from absl import app
from absl import flags
from absl import logging
from collections import OrderedDict
from tensorboardX import SummaryWriter
from shutil import copyfile

import warnings
warnings.simplefilter(action='ignore')

# reproducibility
random.seed(216)
torch.manual_seed(216)
np.random.seed(216)

# set configurations
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', './data/fda_drugs', 'Directory where data is stored')
flags.DEFINE_string('data_file', 'smiles_set_clean.smi', 'Data file name')
flags.DEFINE_string('fda_file', 'all_drugs.smi', 'FDA drugs SMILES file name')
flags.DEFINE_string('vocab_file', 'char_set_clean.pkl', 'Vocabulary file name')
flags.DEFINE_string('atc_sim_file', 'drugs_sp_all.csv', 'ATC drug-drug path distances')
flags.DEFINE_string('checkpoint_dir', './experiments/SMILES', 'Directory where model is stored')
flags.DEFINE_string('experiment_name', 'debug', 'Experiment name')
flags.DEFINE_string('task', 'vae + atc', 'Task(s) included in this experiment')
flags.DEFINE_integer('limit', 5000, 'Training sample size limit')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('max_sequence_length', 120, 'Maximum length of input sequence')
flags.DEFINE_float('learning_rate', 3e-4, 'Initial learning rate')
flags.DEFINE_float('max_norm', 1e3, 'Maximum total gradient norm')
flags.DEFINE_float('wd', 0, 'Weight decay(L2 penalty)')
flags.DEFINE_string('manifold_type', 'Lorentz', 'Latent space type')
flags.DEFINE_string('prior_type', 'Standard', 'Prior type: Standard normal or VampPrior')
flags.DEFINE_integer('num_centroids', 20, 'Number of centroids used in VampPrior')
flags.DEFINE_boolean('bidirectional', False, 'Encoder RNN bidirectional indicator')
flags.DEFINE_integer('num_layers', 3, 'RNN number of layers')
flags.DEFINE_integer('hidden_size', 20, 'Dimension of RNN output')
flags.DEFINE_integer('latent_size', 2, 'Dimension of latent space Z')
flags.DEFINE_float('word_dropout_rate', 0.2, 'Decoder input drop out rate')
flags.DEFINE_string('anneal_function', 'logistic', 'KL annealing function type')
flags.DEFINE_float('k', 0.51, '1st parameter in KL logistic annealing function')
flags.DEFINE_float('x0', 29, '2nd parameter in KL logistic annealing function')
flags.DEFINE_float('C', 1, 'Constant value if KL annealing function is a constant')
flags.DEFINE_integer('num_workers', 2, 'Number of workers in DataLoader')
flags.DEFINE_integer('logging_steps', 1, 'Log per steps/mini-batch')
flags.DEFINE_integer('save_per_epochs', 5, 'Save intermediate checkpoints every few training epochs')
flags.DEFINE_boolean('new_training', True, 'New training or restart from a pre-trained checkpoint')
flags.DEFINE_boolean('new_annealing', True, 'Restart KL annealing from a pre-trained checkpoint')
flags.DEFINE_string('checkpoint', 'checkpoint_epoch010.model', 'Load checkpoint file')
flags.DEFINE_integer('trained_epochs', 10, 'Number of epochs that have been trained')
flags.DEFINE_float('alpha', 0.0, 'Weight of KL divergence between marginal posterior and prior')
flags.DEFINE_float('beta', 0.0, 'Weight of KL divergence between conditional posterior and prior')
flags.DEFINE_float('gamma', 1.0, 'Weight of MMD divergence between conditional posterior and prior')
flags.DEFINE_float('delta', 5.0, 'Weight of ATC local ranking loss')
flags.DEFINE_integer('nneg', 5, 'Number of negative examples sampled when calculating local ranking loss')
flags.DEFINE_float('fda_prop', 0.2, 'Average proportion of FDA drugs in one batch')

def save_and_load_flags():
    """
    save and load experiment configurations
    :return: configuration dict
    """
    experiment_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
    flag_saving_path = os.path.join(experiment_dir, 'configs.json')

    # save model configurations
    if FLAGS.new_training:
        if os.path.isdir(experiment_dir):
            raise ValueError('Experiment directory already exist. Please change experiment name.')
        else:
            os.makedirs(experiment_dir)

        # define model configuration
        configs = {
            'data_dir': FLAGS.data_dir,
            'data_file': FLAGS.data_file,
            'fda_file': FLAGS.fda_file,
            'vocab_file': FLAGS.vocab_file,
            'atc_sim_file': FLAGS.atc_sim_file,
            'checkpoint_dir': FLAGS.checkpoint_dir,
            'experiment_name': FLAGS.experiment_name,
            'task': FLAGS.task,
            'limit': FLAGS.limit,
            'batch_size': FLAGS.batch_size,
            'epochs': FLAGS.epochs,
            'max_sequence_length': FLAGS.max_sequence_length,
            'learning_rate': FLAGS.learning_rate,
            'max_norm': FLAGS.max_norm,
            'wd': FLAGS.wd,
            'manifold_type': FLAGS.manifold_type,
            'prior_type': FLAGS.prior_type,
            'num_centroids': FLAGS.num_centroids,
            'bidirectional': FLAGS.bidirectional,
            'num_layers': FLAGS.num_layers,
            'hidden_size': FLAGS.hidden_size,
            'latent_size': FLAGS.latent_size,
            'word_dropout_rate': FLAGS.word_dropout_rate,
            'anneal_function': FLAGS.anneal_function,
            'k': FLAGS.k,
            'x0': FLAGS.x0,
            'C': FLAGS.C,
            'num_workers': FLAGS.num_workers,
            'logging_steps': FLAGS.logging_steps,
            'save_per_epochs': FLAGS.save_per_epochs,
            'new_training': FLAGS.new_training,
            'new_annealing': FLAGS.new_annealing,
            'checkpoint': FLAGS.checkpoint,
            'trained_epochs': FLAGS.trained_epochs,
            'alpha': FLAGS.alpha,
            'beta': FLAGS.beta,
            'gamma': FLAGS.gamma,
            'delta': FLAGS.delta,
            'nneg': FLAGS.nneg,
            'fda_prop': FLAGS.fda_prop
        }

        with open(flag_saving_path, 'w') as fp:
            json.dump(configs, fp, indent=2)
        fp.close()
    else:
        # overwrite old configurations
        with open(flag_saving_path, 'r') as fp:
            configs = json.load(fp)
            for key, value in configs.items():
                configs[key] = FLAGS[key].value
        fp.close()
        with open(flag_saving_path, 'w') as fp:
            json.dump(configs, fp, indent=2)
        fp.close()

    if not os.path.exists(flag_saving_path):
        raise AssertionError("Training model configuration file did't find.")

    logging.info('Saved model parameters:')
    for i, (key, val) in enumerate(configs.items()):
        logging.info('%d: %s=%s', i, key, val)
    return configs

def create_raw_files(configs, split_proportion):
    """
    create train, valid, test files in the experiment folder
    :param configs:
    :param split_proportion:
    :return:
    """
    # create train, valid, test files
    limit = configs['limit']
    experiment_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
    smiles_file = os.path.join(configs['data_dir'], configs['data_file'])

    with open(smiles_file, 'r') as f:
        smiles = f.readlines()
        random.shuffle(smiles) # random shuffle
        if limit == 0:
            limit = len(smiles)
        train_size = round(limit * split_proportion[0])
        valid_size = round(limit * split_proportion[1])
        smiles_train = smiles[:train_size]
        smiles_valid = smiles[train_size:(train_size + valid_size)]
        smiles_test = smiles[(train_size+valid_size):limit]
    f.close()

    #save a copy in the experiment folder
    f_train = open(os.path.join(experiment_dir, 'smiles_train.smi'), 'w')
    f_train.writelines(smiles_train)
    f_train.close()
    logging.info('Training data has been created. Size = %d.' % (train_size))

    f_valid = open(os.path.join(experiment_dir, 'smiles_valid.smi'), 'w')
    f_valid.writelines(smiles_valid)
    f_valid.close()
    logging.info('Validation data has been created. Size = %d.' % (valid_size))

    f_test = open(os.path.join(experiment_dir, 'smiles_test.smi'), 'w')
    f_test.writelines(smiles_test)
    f_test.close()
    logging.info('Testing data has been created. Size = %d.' % (limit - train_size - valid_size))

    # save a copy of all_drugs.smi
    src = './data/fda_drugs/all_drugs.smi'
    dst = os.path.join(experiment_dir, 'all_drugs.smi')
    copyfile(src, dst)

class MixSampler(Sampler):

    def __init__(self, data_source_1, data_source_2, prop, num_samples=None):
        self.data_source_1 = data_source_1
        self.data_source_2 = data_source_2
        self.prop = prop
        self.num_samples = num_samples
        self.num_samples_1 = int(self.num_samples * prop)
        self.num_samples_2 = self.num_samples - self.num_samples_1

    def __iter__(self):
        samples_idx = []
        samples_idx = samples_idx + np.random.choice(self.data_source_1, self.num_samples_1).tolist()
        samples_idx = samples_idx + np.random.choice(self.data_source_2, self.num_samples_2, replace=False).tolist()
        random.shuffle(samples_idx)
        return iter(samples_idx)

    def __len__(self):
        return self.num_samples


def pipeline(configs):

    # step 1: prepare datasets for dataloader
    experiment_dir = os.path.join(configs['checkpoint_dir'], configs['experiment_name'])
    datasets = OrderedDict()

    # prepare FDA only dataset
    fda_dataset = drugdata(task=configs['task'],
                       fda_drugs_dir=configs['data_dir'],
                       fda_smiles_file=configs['fda_file'],
                       fda_vocab_file=configs['vocab_file'],
                       fda_drugs_sp_file=configs['atc_sim_file'],
                       experiment_dir=experiment_dir,
                       smi_file=configs['fda_file'],
                       max_sequence_length=configs['max_sequence_length'],
                       nneg=configs['nneg']
                       )

    # prepare train, valid, test datasets
    splits = ['train', 'valid', 'test']
    for split in splits:
        datasets[split] = drugdata(task = configs['task'],
                                   fda_drugs_dir=configs['data_dir'],
                                   fda_smiles_file=configs['fda_file'],
                                   fda_vocab_file=configs['vocab_file'],
                                   fda_drugs_sp_file=configs['atc_sim_file'],
                                   experiment_dir=experiment_dir,
                                   smi_file='smiles_' + split + '.smi',
                                   max_sequence_length=configs['max_sequence_length'],
                                   nneg=configs['nneg'])

    # create train sampler to ensure FDA drug sampling rate ~ configs['fda_prop'] in a batch
    Mixsampler = MixSampler(datasets['train'].fda_idx, datasets['train'].zinc_idx, configs['fda_prop'], len(datasets['train']))

    # step 2: define model
    if configs['manifold_type'] == 'Euclidean':
        model = EVAE(
            vocab_size=datasets['train'].vocab_size,
            hidden_size=configs['hidden_size'],
            latent_size=configs['latent_size'],
            bidirectional=configs['bidirectional'],
            num_layers=configs['num_layers'],
            word_dropout_rate=configs['word_dropout_rate'],
            max_sequence_length=configs['max_sequence_length'],
            sos_idx=datasets['train'].sos_idx,
            eos_idx=datasets['train'].eos_idx,
            pad_idx=datasets['train'].pad_idx,
            unk_idx=datasets['train'].unk_idx,
            prior=configs['prior_type'],
            alpha=configs['alpha'],
            beta=configs['beta'],
            gamma=configs['gamma']
        )
    elif configs['manifold_type'] == 'Lorentz':
        model = HVAE(
            vocab_size=datasets['train'].vocab_size,
            hidden_size=configs['hidden_size'],
            latent_size=configs['latent_size'],
            bidirectional=configs['bidirectional'],
            num_layers=configs['num_layers'],
            word_dropout_rate=configs['word_dropout_rate'],
            max_sequence_length=configs['max_sequence_length'],
            sos_idx=datasets['train'].sos_idx,
            eos_idx=datasets['train'].eos_idx,
            pad_idx=datasets['train'].pad_idx,
            unk_idx=datasets['train'].unk_idx,
            prior=configs['prior_type'],
            alpha=configs['alpha'],
            beta=configs['beta'],
            gamma=configs['gamma']
        )

    # load checkpoint
    if configs['new_training'] is False:
        checkpoint_path = os.path.join(configs['checkpoint_dir'] + '/' + configs['experiment_name'], configs['checkpoint'])
        model.load_state_dict(torch.load(checkpoint_path))
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()
    print(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['wd'])

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # step 3: start training
    summary_writer = SummaryWriter(log_dir=experiment_dir)
    logging.info('Initialization complete. Start training.')

    try:
        if configs['new_training']:
            start_epoch = 1
            end_epoch = configs['epochs']
        else:
            start_epoch = configs['trained_epochs'] + 1
            end_epoch = start_epoch + configs['epochs'] - 1

        step = 0 # SGD updates
        for epoch in range(start_epoch, end_epoch + 1):
            for split in splits:

                # prepare dataloader
                if split == 'train':
                    if configs['data_file'] == 'smiles_set_clean.smi': # if dataset = ZINC + FDA
                        DrugsLoader = DataLoader(
                            dataset=datasets[split],
                            batch_size=configs['batch_size'],
                            sampler=Mixsampler,
                            drop_last=True,
                            num_workers=configs['num_workers'],
                            pin_memory=torch.cuda.is_available()
                        )
                    elif configs['data_file'] == 'all_drugs.smi': # if dataset = FDA
                        DrugsLoader = DataLoader(
                            dataset=datasets[split],
                            batch_size=configs['batch_size'],
                            drop_last=False,
                            num_workers=configs['num_workers'],
                            pin_memory=torch.cuda.is_available()
                        )
                else: # split = valida or test
                    DrugsLoader = DataLoader(
                        dataset=datasets[split],
                        batch_size=configs['batch_size'],
                        shuffle=False,
                        drop_last=False,
                        num_workers=configs['num_workers'],
                        pin_memory=torch.cuda.is_available()
                    )

                # define performance metrics
                epoch_metric = {
                    'total_loss': [],
                    'recon_loss': [],
                    'kl_loss': [],
                    'mkl_loss': [],
                    'mmd_loss': [],
                    'kl_weight': [],
                    'local_ranking_loss': []
                }

                for iteration, batch in enumerate(DrugsLoader):

                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = to_cuda_var(v)

                    recon_loss, kl_loss, mkl_loss, mmd_loss, local_ranking_loss = model(configs['task'], batch, len(DrugsLoader.dataset)) # forward pass and compute losses

                    anneal_weight = kl_anneal_function(configs, start_epoch, epoch)
                    loss = (1.0 * recon_loss
                            + anneal_weight * (configs['beta'] * kl_loss + configs['alpha'] * mkl_loss + configs['gamma'] * mmd_loss)
                            + configs['delta'] * local_ranking_loss)

                    # initialize performance metrics
                    epoch_metric['total_loss'].append(loss.item())
                    epoch_metric['recon_loss'].append(recon_loss.item())
                    epoch_metric['kl_loss'].append(kl_loss.item())
                    epoch_metric['mkl_loss'].append(mkl_loss.item())
                    epoch_metric['mmd_loss'].append(mmd_loss.item())
                    epoch_metric['kl_weight'].append(anneal_weight)
                    if local_ranking_loss > 0: # samples with ATC information
                        epoch_metric['local_ranking_loss'].append(local_ranking_loss.item())

                    if split == 'train':
                        # backprop and gradient descent
                        optimizer.zero_grad()
                        loss.backward()

                        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs
                        torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_norm'])

                        optimizer.step()
                        step += 1

                        # track model gradients norm
                        #(total_grad_norm, enc_grad_norm, dec_grad_norm, h2m_grad_norm, h2v_grad_norm) = track_gradient_norm(model)

                        logging.info('Training: Epoch [{}/{}], Batch [{}/{}], RECON-Loss: {:.4f}, KL-Loss: {:.4f}, MKL-Loss: {:.4f}, MMD-Loss: {:.4f}, KL-Annealing: {:.4f}, RANK-Loss: {:.4f}'
                            .format(epoch, end_epoch, iteration + 1, len(DrugsLoader), recon_loss.item(), kl_loss.item(), mkl_loss.item(), mmd_loss.item(), anneal_weight, local_ranking_loss.item()))

                        # saving model checkpoint
                        if (epoch) % configs['save_per_epochs'] == 0 and iteration + 1 == len(DrugsLoader):
                            logging.info('Saving model checkpoint...')
                            checkpoint_name = os.path.join(experiment_dir, 'checkpoint_epoch%03d.model' % (epoch))
                            torch.save(model.state_dict(), checkpoint_name)

                    elif split == 'valid':
                        #TODO: early stopping metric
                        pass


                # tensorboardX tracker
                summary_writer.add_scalar('%s/avg_total_loss' % split, np.array(epoch_metric['total_loss']).mean(), epoch)
                summary_writer.add_scalar('%s/avg_recon_loss' % split, np.array(epoch_metric['recon_loss']).mean(), epoch)
                summary_writer.add_scalar('%s/avg_kl_loss' % split, np.array(epoch_metric['kl_loss']).mean(), epoch)
                summary_writer.add_scalar('%s/avg_mkl_loss' % split, np.array(epoch_metric['mkl_loss']).mean(), epoch)
                summary_writer.add_scalar('%s/avg_mmd_loss' % split, np.array(epoch_metric['mmd_loss']).mean(), epoch)
                summary_writer.add_scalar('%s/avg_kl_weight' % split, np.array(epoch_metric['kl_weight']).mean(), epoch)
                summary_writer.add_scalar('%s/avg_local_ranking_loss' % split, np.array(epoch_metric['local_ranking_loss']).mean(), epoch)

                # compute dendrogram purity
                if split == 'train' and (epoch % configs['save_per_epochs'] == 0):
                    # DP on training
                    start_time = time.time()
                    drug_lst, mean_lst, _ = fda_drug_rep(configs, datasets[split], model, all_drugs=False)
                    dp_lvl4 = dendrogram_purity_score(configs, drug_lst, mean_lst, atc_lvl=4)
                    dp_lvl3 = dendrogram_purity_score(configs, drug_lst, mean_lst, atc_lvl=3)
                    dp_lvl2 = dendrogram_purity_score(configs, drug_lst, mean_lst, atc_lvl=2)
                    dp_lvl1 = dendrogram_purity_score(configs, drug_lst, mean_lst, atc_lvl=1)
                    end_time = time.time()
                    summary_writer.add_scalar('%s/puprity_lvl4' % split, np.array(dp_lvl4), epoch)
                    summary_writer.add_scalar('%s/puprity_lvl3' % split, np.array(dp_lvl3), epoch)
                    summary_writer.add_scalar('%s/puprity_lvl2' % split, np.array(dp_lvl2), epoch)
                    summary_writer.add_scalar('%s/puprity_lvl1' % split, np.array(dp_lvl1), epoch)
                    logging.info(
                        'Training DP: Epoch [{}/{}], DP ATC Lvl 1: {:.4f}, DP ATC Lvl2: {:.4f}, DP ATC Lvl3: {:.4f}, DP ATC Lvl4: {:.4f}, Time: {:.2f}'
                        .format(epoch, end_epoch, dp_lvl1, dp_lvl2, dp_lvl3, dp_lvl4, (end_time-start_time)))


                    # DP on FDA
                    start_time = time.time()
                    drug_lst, mean_lst, _ = fda_drug_rep(configs, fda_dataset, model, all_drugs=True)
                    dp_lvl4 = dendrogram_purity_score(configs, drug_lst, mean_lst, atc_lvl=4)
                    dp_lvl3 = dendrogram_purity_score(configs, drug_lst, mean_lst, atc_lvl=3)
                    dp_lvl2 = dendrogram_purity_score(configs, drug_lst, mean_lst, atc_lvl=2)
                    dp_lvl1 = dendrogram_purity_score(configs, drug_lst, mean_lst, atc_lvl=1)
                    end_time = time.time()
                    summary_writer.add_scalar('FDA/puprity_lvl4', np.array(dp_lvl4), epoch)
                    summary_writer.add_scalar('FDA/puprity_lvl3', np.array(dp_lvl3), epoch)
                    summary_writer.add_scalar('FDA/puprity_lvl2', np.array(dp_lvl2), epoch)
                    summary_writer.add_scalar('FDA/puprity_lvl1', np.array(dp_lvl1), epoch)
                    logging.info(
                        'FDA DP: Epoch [{}/{}], DP ATC Lvl 1: {:.4f}, DP ATC Lvl2: {:.4f}, DP ATC Lvl3: {:.4f}, DP ATC Lvl4: {:.4f}, Time: {:.2f}'
                        .format(epoch, end_epoch, dp_lvl1, dp_lvl2, dp_lvl3, dp_lvl4, (end_time-start_time)))


    except KeyboardInterrupt:
        logging.info('Interrupted! Stop Training!')

        logging.info('Saving model checkpoint...')
        checkpoint_name = os.path.join(experiment_dir, 'checkpoint_epoch%03d_batch%03d.model' % (epoch, iteration + 1))
        torch.save(model.state_dict(), checkpoint_name)
        logging.info('Model saved at %s' % checkpoint_name)

    finally:
        logging.info('Training completed.')


def main(args):
    del args

    # get model configurations
    configs = save_and_load_flags()

    # create train, valid, test data
    splits_proportion = [0.9, 0.05, 0.05]
    create_raw_files(configs, splits_proportion)

    # start pipeline
    pipeline(configs)

if __name__=='__main__':
    app.run(main)