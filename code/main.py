import sys

from textdata import textdata
from model import *

from absl import app
from absl import flags
from absl import logging
from tensorboardX import SummaryWriter

from collections import OrderedDict
from torch.utils.data import DataLoader
import torch
import time
import datetime

now = datetime.datetime.now()

# define model hyper parameters
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', './data', 'Directory where data is stored')
flags.DEFINE_string('data_file','all_drugs_only.smi', 'Data file name')
flags.DEFINE_string('vocab_file', 'zinc_char_list.json', 'Vocabulary file name')
flags.DEFINE_string('checkpoint_dir', './experiments', 'Directory where model is stored')
flags.DEFINE_string('experiment_name', 'debug', 'Experiment name')
flags.DEFINE_integer('limit', 1396, 'Training sample size limit')
flags.DEFINE_integer('batch_size', 16, 'Mini batch size')
flags.DEFINE_integer('epochs', 50, 'Number of epochs')
flags.DEFINE_integer('max_sequence_length', 120, 'Maximum length of input sequence')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate')
flags.DEFINE_string('manifold_type', 'Euclidean', 'Latent space type')
flags.DEFINE_string('rnn_type', 'gru', 'RNN type')
flags.DEFINE_boolean('bidirectional', False, 'RNN bidirectional indicator')
flags.DEFINE_integer('num_layers', 1, 'RNN number of layers')
flags.DEFINE_integer('hidden_size', 488, 'Dimension of RNN output')
flags.DEFINE_integer('latent_size', 2, 'Dimension of latent space Z')
flags.DEFINE_boolean('one_hot_rep', True, 'Use one hot vector to represent inputs or not')
flags.DEFINE_float('word_dropout_rate', 0.25, 'Decoder input drop out rate')
flags.DEFINE_float('embedding_dropout_rate', 0.2, 'Embedding drop out rate')
flags.DEFINE_string('anneal_function', 'logistic', 'KL annealing function type')
flags.DEFINE_float('k', 0.0025, '1st parameter in KL logistic annealing function')
flags.DEFINE_float('x0', 2500, '2nd parameter in KL logistic annealing function')
flags.DEFINE_integer('num_workers', 1, 'Number of workers in DataLoader')
flags.DEFINE_integer('logging_steps', 25, 'Log per steps/mini-batch')
flags.DEFINE_integer('save_per_epochs', 10, 'Save intermediate checkpoints every few training epochs')


def save_and_load_flags():
    experiment_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
    flag_saving_path = os.path.join(experiment_dir, 'configs.json')

    # save model configurations
    if os.path.isdir(experiment_dir):
        raise ValueError('Experiment directory already exist. Please change experiment name.')
    else:
        os.makedirs(experiment_dir)

        # define model configuration
        configs = {
            'data_dir': FLAGS.data_dir,
            'data_file': FLAGS.data_file,
            'vocab_file': FLAGS.vocab_file,
            'checkpoint_dir': FLAGS.checkpoint_dir,
            'experiment_name': FLAGS.experiment_name,
            'limit': FLAGS.limit,
            'batch_size': FLAGS.batch_size,
            'epochs': FLAGS.epochs,
            'max_sequence_length': FLAGS.max_sequence_length,
            'learning_rate': FLAGS.learning_rate,
            'manifold_type': FLAGS.manifold_type,
            'rnn_type': FLAGS.rnn_type,
            'bidirectional': FLAGS.bidirectional,
            'num_layers': FLAGS.num_layers,
            'hidden_size': FLAGS.hidden_size,
            'latent_size': FLAGS.latent_size,
            'one_hot_rep': FLAGS.one_hot_rep,
            'word_dropout_rate': FLAGS.word_dropout_rate,
            'embedding_dropout_rate': FLAGS.embedding_dropout_rate,
            'anneal_function': FLAGS.anneal_function,
            'k': FLAGS.k,
            'x0': FLAGS.x0,
            'num_workers': FLAGS.num_workers,
            'logging_steps': FLAGS.logging_steps,
            'save_per_epochs': FLAGS.save_per_epochs
        }

        with open(flag_saving_path, 'w') as fp:
            json.dump(configs, fp, indent=2)

    if not os.path.exists(flag_saving_path):
        raise AssertionError("Training model configuration file did't find.")

    logging.info('Saved model parameters:')
    for i, (key, val) in enumerate(configs.items()):
        logging.info('%d: %s=%s', i, key, val)

    return configs


def create_raw_files(configs, split_proportion):
    # create train, valid, test files
    limit = configs['limit']
    data_dir = configs['data_dir']
    smiles_file = os.path.join(configs['data_dir'], configs['data_file'])

    with open(smiles_file, 'r') as f:
        smiles = f.readlines()
        if limit is None:
            limit = len(smiles)
        train_size = round(limit * split_proportion[0])
        valid_size = round(limit * split_proportion[1])
        smiles_train = smiles[:train_size]
        smiles_valid = smiles[train_size:(train_size + valid_size)]
        smiles_test = smiles[(train_size+valid_size):limit]
    f.close()


    cnt_cvrt_cans=0

    f_train = open(os.path.join(data_dir, 'smiles_train.smi'), 'w')
    #smiles_train, cnt_cvrt_cans = check_canonical_form(smiles_train)
    f_train.writelines(smiles_train)
    f_train.close()
    logging.info('Training data has been created. Size = %d. Converted %d non-canonical SMILES.' % (train_size, cnt_cvrt_cans))

    f_valid = open(os.path.join(data_dir, 'smiles_valid.smi'), 'w')
    #smiles_valid, cnt_cvrt_cans = check_canonical_form(smiles_valid)
    f_valid.writelines(smiles_valid)
    f_valid.close()
    logging.info('Validation data has been created. Size = %d. Converted %d non-canonical SMILES.' % (valid_size, cnt_cvrt_cans))

    f_test = open(os.path.join(data_dir, 'smiles_test.smi'), 'w')
    #smiles_test, cnt_cvrt_cans = check_canonical_form(smiles_test)
    f_test.writelines(smiles_test)
    f_test.close()
    logging.info('Testing data has been created. Size = %d. Converted %d non-canonical SMILES.' % (limit - train_size - valid_size, cnt_cvrt_cans))
    return True


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)
    elif anneal_function == 'constant':
        return 1

def kl_lorentz(z, mean, logv):
    [batch_size, n_h] = mean.shape
    n = n_h - 1
    mu0 = to_cuda_var(torch.zeros(batch_size, n))
    mu0_h = lorentz_mapping_origin(mu0)
    diag = to_cuda_var(torch.eye(n).repeat(batch_size, 1, 1))
    # sampling z at mu_h on hyperbolic space
    cov = torch.exp(logv).unsqueeze(dim=2) * diag
    # posterior density
    _, logp_z = pseudo_hyperbolic_gaussian(z, mean, cov, version=2)
    # prior density
    _, logp_z0 = pseudo_hyperbolic_gaussian(z, mu0_h, diag, version=2)
    return torch.sum(logp_z - logp_z0)

def loss_fn(NLL, logp, target, length, z, mean, logv, anneal_function, step, k, x0, manifold_type):
    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    if manifold_type == 'Euclidean':
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    elif manifold_type == 'Lorentz':
        KL_loss = kl_lorentz(z, mean, logv)

    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

#TODO restart training from a checkpoint
def pipeline(configs):
    root_dir = os.path.join(configs['checkpoint_dir'], configs['experiment_name'])

    with open(os.path.join(root_dir, "running_configs.cfg"), "w") as fp:
        fp.write("python ")
        for i, x in enumerate(sys.argv):
            logging.info("%d: %s", i, x)
            fp.write("%s \\\n" % x)

    # prepare train, valid, test datasets
    datasets = OrderedDict()
    splits = ['train', 'valid', 'test']
    for split in splits:
        datasets[split] = textdata(directory=configs['data_dir'], split=split, vocab_file=configs['vocab_file'],
                                   max_sequence_length=configs['max_sequence_length'])

    # build model
    model = MolVAE(
        vocab_size=datasets['train'].vocab_size,
        embedding_size=datasets['train'].vocab_size,
        hidden_size=configs['hidden_size'],
        latent_size=configs['latent_size'],
        manifold_type=configs['manifold_type'],
        rnn_type=configs['rnn_type'],
        bidirectional=configs['bidirectional'],
        num_layers=configs['num_layers'],
        word_dropout_rate=configs['word_dropout_rate'],
        embedding_dropout_rate=configs['embedding_dropout_rate'],
        one_hot_rep=configs['one_hot_rep'],
        max_sequence_length=configs['max_sequence_length'],
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx
    )

    # TODO: GPU
    if torch.cuda.is_available():
        model = model.cuda()
        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    print(model)

    # reconstruction loss
    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])

    # learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # tensorboard tracker
    summary_writer = SummaryWriter(log_dir=root_dir)
    logging.info('Initialization complete. Start training.')

    step = 0
    try:
        for epoch in range(configs['epochs']):

            for split in splits:

                data_loader = DataLoader(
                    dataset=datasets[split],
                    batch_size=configs['batch_size'],
                    shuffle=(split=='train'),
                    num_workers=8,
                    pin_memory=torch.cuda.is_available()
                )

                epoch_metric = {
                    'total_loss': [],
                    'nll_loss': [],
                    'kl_loss': [],
                    'kl_weight': [],
                    'lr':[]
                }

                start_time = time.time()
                for iteration, batch in enumerate(data_loader):
                    # current batch size which can be smaller than default batch size when it is the last batch
                    batch_size = batch['inputs'].size(0)

                    # TODO: GPU
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = to_cuda_var(v)

                    # forward
                    logp, mean, logv, z = model(batch['inputs'], batch['len'])

                    # loss calculation
                    NLL_loss, KL_loss, KL_weight = loss_fn(NLL, logp, batch['targets'],
                        batch['len'], z, mean, logv, configs['anneal_function'], step, configs['k'], configs['x0'], configs['manifold_type'])

                    loss = (NLL_loss + KL_weight * KL_loss)/batch_size

                    #record metrics
                    epoch_metric['total_loss'].append(loss.item())
                    epoch_metric['nll_loss'].append(NLL_loss.item()/batch_size)
                    epoch_metric['kl_loss'].append(KL_loss.item()/batch_size)
                    epoch_metric['kl_weight'].append(KL_weight)
                    epoch_metric['lr'].append((optimizer.param_groups[0]['lr']))

                    # backward + optimization
                    if split == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        step += 1

                        # logging training status
                        if iteration == 0 or (iteration + 1) % configs['logging_steps'] == 0 or iteration + 1 == len(data_loader):
                            end_time = time.time()
                            logging.info(
                                'Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, NLL-Loss: {:.4f}, KL-Loss: {:.4f}, KL-Weight: {:.4f}, Learning Rate: {:.6f}, Speed: {:4f} ms'.
                                format(epoch + 1, configs['epochs'], iteration + 1, len(data_loader), loss.item(), NLL_loss.item()/batch_size, KL_loss.item()/batch_size, KL_weight, optimizer.param_groups[0]['lr'],
                                       (end_time - start_time) * 100 / batch_size))
                            start_time = time.time()

                        # saving model checkpoint
                        if (epoch + 1) % configs['save_per_epochs'] == 0 and iteration + 1 == len(data_loader):
                            logging.info('Saving model checkpoint...')
                            checkpoint_name = os.path.join(root_dir, 'checkpoint_epoch%03d.model' % (epoch + 1))
                            torch.save(model.state_dict(), checkpoint_name)

                        # training + batch level
                        #batch_update_metrics(summary_writer, split, step, loss.item(), NLL_loss.item() / batch_size, KL_loss.item() / batch_size, KL_weight, optimizer.param_groups[0]['lr'])

                # split (e.g. train, valid, test) + epoch level
                epoch_update_metrics(summary_writer, split, epoch, epoch_metric['total_loss'], epoch_metric['nll_loss'], epoch_metric['kl_loss'], epoch_metric['kl_weight'], epoch_metric['lr'])

    except KeyboardInterrupt:
        logging.info('Interrupted! Stop Training!')

        logging.info('Saving model checkpoint...')
        checkpoint_name = os.path.join(root_dir, 'checkpoint_epoch%03d_batch%03d.model' % (epoch + 1, iteration + 1))
        torch.save(model.state_dict(), checkpoint_name)
        logging.info('Model saved at %s' % checkpoint_name)

    finally:
        logging.info('Training completed.')

    # save final model
    #save model
    endpoint_name = os.path.join(root_dir, 'endpoint_%04d%02d%02d.model' % (int(now.year), int(now.month), int(now.day)))
    torch.save(model, endpoint_name)

    return




def main(argv):
    del argv

    # get model configurations
    configs = save_and_load_flags()

    # create train, valid, test data
    splits_proportion = [0.8, 0.1, 0.1]
    create_raw_files(configs, splits_proportion)

    # start pipeline
    pipeline(configs)



# start training
if __name__ == '__main__':
    app.run(main)