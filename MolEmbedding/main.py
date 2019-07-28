from textdata import textdata
from model import *
from utils import *

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
flags.DEFINE_string('data_file','all_drugs_zinc.smi', 'Data file name')
flags.DEFINE_string('vocab_file', 'zinc_char_list.json', 'Vocabulary file name')
flags.DEFINE_string('checkpoint_dir', './experiments/SMILES', 'Directory where model is stored')
flags.DEFINE_string('experiment_name', 'debug', 'Experiment name')
flags.DEFINE_integer('limit', 400, 'Training sample size limit')
flags.DEFINE_integer('batch_size', 32, 'Mini batch size')
flags.DEFINE_integer('epochs', 20, 'Number of epochs')
flags.DEFINE_integer('max_sequence_length', 120, 'Maximum length of input sequence')
flags.DEFINE_float('learning_rate', 3e-4, 'Initial learning rate')
flags.DEFINE_float('max_norm', 1e12, 'Maximum total graident norm')
flags.DEFINE_float('wd', 0, 'Weight decay(L2 penalty)')
flags.DEFINE_string('manifold_type', 'Lorentz', 'Latent space type')
flags.DEFINE_string('rnn_type', 'gru', 'RNN type')
flags.DEFINE_boolean('bidirectional', True, 'RNN bidirectional indicator')
flags.DEFINE_integer('num_layers', 2, 'RNN number of layers')
flags.DEFINE_integer('hidden_size', 50, 'Dimension of RNN output')
flags.DEFINE_integer('latent_size', 5, 'Dimension of latent space Z')
flags.DEFINE_boolean('one_hot_rep', True, 'Use one hot vector to represent inputs or not')
flags.DEFINE_float('word_dropout_rate', 0.2, 'Decoder input drop out rate')
flags.DEFINE_float('embedding_dropout_rate', 0.0, 'Embedding drop out rate')
flags.DEFINE_string('anneal_function', 'logistic', 'KL annealing function type')
flags.DEFINE_float('k', 0.51, '1st parameter in KL logistic annealing function')
flags.DEFINE_float('x0', 29, '2nd parameter in KL logistic annealing function')
flags.DEFINE_integer('num_workers', 1, 'Number of workers in DataLoader')
flags.DEFINE_integer('logging_steps', 1, 'Log per steps/mini-batch')
flags.DEFINE_integer('save_per_epochs', 5, 'Save intermediate checkpoints every few training epochs')
flags.DEFINE_boolean('new_training', True, 'New training or restart from a pretrained checkpoint')
flags.DEFINE_boolean('new_annealing', False, 'Restart KL annealing from a pretrained checkpoint')
flags.DEFINE_string('checkpoint', 'checkpoint_epoch005.model', 'Load checkpoint file')
flags.DEFINE_integer('trained_epochs', 5, 'Number of epochs that have been trained')
flags.DEFINE_float('prior_var', 1.0, 'Variance of prior distribution')
flags.DEFINE_float('beta', 1.0, 'Weight of KL divergence between conditional posterior and prior')
flags.DEFINE_float('alpha', 0.0, 'Weight of KL divergence between marginal posterior and prior')


def save_and_load_flags():
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
            'vocab_file': FLAGS.vocab_file,
            'checkpoint_dir': FLAGS.checkpoint_dir,
            'experiment_name': FLAGS.experiment_name,
            'limit': FLAGS.limit,
            'batch_size': FLAGS.batch_size,
            'epochs': FLAGS.epochs,
            'max_sequence_length': FLAGS.max_sequence_length,
            'learning_rate': FLAGS.learning_rate,
            'max_norm': FLAGS.max_norm,
            'wd': FLAGS.wd,
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
            'save_per_epochs': FLAGS.save_per_epochs,
            'new_training': FLAGS.new_training,
            'new_annealing': FLAGS.new_annealing,
            'checkpoint': FLAGS.checkpoint,
            'trained_epochs': FLAGS.trained_epochs,
            'prior_var': FLAGS.prior_var,
            'beta': FLAGS.beta,
            'alpha': FLAGS.alpha
        }

        with open(flag_saving_path, 'w') as fp:
            json.dump(configs, fp, indent=2)
        fp.close()

    else:
        with open(flag_saving_path, 'r') as fp:
            configs = json.load(fp)
            configs['new_training'] = FLAGS.new_training
            configs['new_annealing'] = FLAGS.new_annealing
            configs['checkpoint'] = FLAGS.checkpoint
            configs['trained_epochs'] = FLAGS.trained_epochs
        fp.close()

    if not os.path.exists(flag_saving_path):
        raise AssertionError("Training model configuration file did't find.")

    logging.info('Saved model parameters:')
    for i, (key, val) in enumerate(configs.items()):
        logging.info('%d: %s=%s', i, key, val)

    return configs


def create_raw_files(configs, split_proportion):

    if configs['new_training']:

        # create train, valid, test files
        limit = configs['limit']
        data_dir = configs['data_dir']
        experiment_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
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
        #save a copy in the experiment folder
        f_train = open(os.path.join(experiment_dir, 'smiles_train.smi'), 'w')
        #smiles_train, cnt_cvrt_cans = check_canonical_form(smiles_train)
        f_train.writelines(smiles_train)
        f_train.close()
        logging.info('Training data has been created. Size = %d. Converted %d non-canonical SMILES.' % (train_size, cnt_cvrt_cans))

        f_valid = open(os.path.join(data_dir, 'smiles_valid.smi'), 'w')
        #smiles_valid, cnt_cvrt_cans = check_canonical_form(smiles_valid)
        f_valid.writelines(smiles_valid)
        f_valid.close()
        f_valid = open(os.path.join(experiment_dir, 'smiles_valid.smi'), 'w')
        #smiles_valid, cnt_cvrt_cans = check_canonical_form(smiles_valid)
        f_valid.writelines(smiles_valid)
        f_valid.close()
        logging.info('Validation data has been created. Size = %d. Converted %d non-canonical SMILES.' % (valid_size, cnt_cvrt_cans))

        f_test = open(os.path.join(data_dir, 'smiles_test.smi'), 'w')
        #smiles_test, cnt_cvrt_cans = check_canonical_form(smiles_test)
        f_test.writelines(smiles_test)
        f_test.close()
        f_test = open(os.path.join(experiment_dir, 'smiles_test.smi'), 'w')
        #smiles_test, cnt_cvrt_cans = check_canonical_form(smiles_test)
        f_test.writelines(smiles_test)
        f_test.close()
        logging.info('Testing data has been created. Size = %d. Converted %d non-canonical SMILES.' % (limit - train_size - valid_size, cnt_cvrt_cans))

    return True

# estimate the KL divergence between marginal posterior to prior in a batch
def marginal_posterior_divergence(vt, u, z, mean, logv, num_samples, manifold_type, prior_var):
    [batch_size, n_h] = mean.shape

    mu0 = to_cuda_var(torch.zeros(1, n_h-1))
    mu0_h = lorentz_mapping_origin(mu0)
    diag0 = to_cuda_var(torch.eye(n_h-1).repeat(1, 1, 1))

    logq_zb_lst = []
    logp_zb_lst = []
    for b in range(batch_size):
        vt_b = vt[b,:].unsqueeze(0)
        u_b = u[b,:].unsqueeze(0)
        zb = z[b,:].unsqueeze(0)
        mu_b = mean[b,:].unsqueeze(0)
        logv_b = logv[b,:].unsqueeze(0)
        diag_b= to_cuda_var(torch.eye(n_h-1).repeat(1, 1, 1))
        cov_b = torch.exp(logv_b).unsqueeze(dim=2) * diag_b

        # removing b-th mean and logv
        vt_r = vt_b.repeat(batch_size-1,1)
        u_r = u_b.repeat(batch_size-1,1)
        zr = zb.repeat(batch_size-1,1)
        mu_r = torch.cat((mean[:b,:],mean[b+1:,:]))
        logv_r = torch.cat((logv[:b,:],logv[b+1:,:]))
        diag_r = to_cuda_var(torch.eye(n_h - 1).repeat(batch_size-1, 1, 1))
        cov_r = torch.exp(logv_r).unsqueeze(dim=2) * diag_r

        if manifold_type == 'Euclidean':
            pass
        elif manifold_type == 'Lorentz':

            # E[log q(zb)] = - H(q(z))
            _, logq_zb_xb = pseudo_hyperbolic_gaussian(zb, mu_b, cov_b, version=2, vt=vt_b, u=u_b)
            _, logq_zb_xr = pseudo_hyperbolic_gaussian(zr, mu_r, cov_r, version=2, vt=vt_r, u=u_r)

            yb1 = logq_zb_xb - torch.log(to_cuda_var(torch.tensor(num_samples).float()))
            yb2 = logq_zb_xr + torch.log(to_cuda_var(torch.tensor((num_samples-1)/((batch_size-1)*num_samples)).float()))
            yb = torch.cat([yb1, yb2], dim=0)
            logq_zb = torch.logsumexp(yb, dim=0)

            # E[log p(zb)]
            _, logp_zb = pseudo_hyperbolic_gaussian(zb, mu0_h, prior_var * diag0, version=2, vt=None, u=None)

            logq_zb_lst.append(logq_zb)
            logp_zb_lst.append(logp_zb)

    logq_zb = torch.stack(logq_zb_lst, dim=0)
    logp_zb = torch.stack(logp_zb_lst, dim=0).squeeze(-1)

    mpd = (logq_zb - logp_zb).sum()

    return logq_zb, logp_zb, mpd


def kl_anneal_function(anneal_function, new_training, new_annealing, step, start_epoch, epoch, k, x0):

    if new_training is False and new_annealing is True:
        epoch = epoch - start_epoch

    if anneal_function == 'logistic':
        #return float(1 / (1 + np.exp(-k * (step - x0))))
        return float(1 / (1 + np.exp(-k * (epoch - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)
    elif anneal_function == 'constant':
        return 0

def kl_lorentz(z, vt, u, mean, logv, prior_var):
    [batch_size, n_h] = mean.shape
    n = n_h - 1
    mu0 = to_cuda_var(torch.zeros(batch_size, n))
    mu0_h = lorentz_mapping_origin(mu0)
    diag = to_cuda_var(torch.eye(n).repeat(batch_size, 1, 1))
    # sampling z at mu_h on hyperbolic space
    cov = torch.exp(logv).unsqueeze(dim=2) * diag
    # posterior density
    _, logp_z = pseudo_hyperbolic_gaussian(z, mean, cov, version=2, vt=vt, u=u)
    # prior density
    _, logp_z0 = pseudo_hyperbolic_gaussian(z, mu0_h, prior_var * diag, version=2, vt=None, u=None)
    return logp_z, logp_z0, torch.sum(logp_z - logp_z0)

def loss_fn(configs, NLL, logp, target, length, z, vt, u, mean, logv, step, epoch, start_epoch, num_samples):

    anneal_function = configs['anneal_function']
    k = configs['k']
    x0 = configs['x0']
    new_training = configs['new_training']
    new_annealing = configs['new_annealing']
    manifold_type = configs['manifold_type']
    prior_var = to_cuda_var(torch.tensor(configs['prior_var']))

    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL divergence
    if manifold_type == 'Euclidean':
        #KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_loss = -0.5 * torch.sum(1 + logv - prior_var.log() - (logv.exp() + mean.pow(2))/prior_var)
    elif manifold_type == 'Lorentz':
        logq_zx_kl, logp_z0_kl, KL_loss = kl_lorentz(z, vt, u, mean, logv, prior_var)

    KL_weight = kl_anneal_function(anneal_function, new_training, new_annealing, step, start_epoch, epoch, k, x0)

    # marginal posterior divergence
    if configs['alpha'] > 0:
        logq_zb, logp_zb, mpd = marginal_posterior_divergence(vt, u, z, mean, logv, num_samples, manifold_type, prior_var)
    else:
        mpd = to_cuda_var(torch.tensor(0.0))

    return NLL_loss, KL_loss, KL_weight, mpd


def pipeline(configs):

    # fix seed (hyper-parameters tuning)
    seed = 0
    torch.manual_seed(seed)

    # experiment root directory
    root_dir = os.path.join(configs['checkpoint_dir'], configs['experiment_name'])

    # save configurations
    with open(os.path.join(root_dir, "running_configs.cfg"), "w") as fp:
        fp.write("python ")
        for i, x in enumerate(sys.argv):
            logging.info("%d: %s", i, x)
            fp.write("%s \\\n" % x)

    # prepare validation SMILES lists
    smiles_fda = create_smiles_lst('./data', 'all_drugs_only.smi')
    smiles_test = create_smiles_lst(configs['checkpoint_dir'] + '/' + configs['experiment_name'], 'smiles_test.smi')

    # prepare train, valid, test datasets
    datasets = OrderedDict()
    splits = ['train', 'valid', 'test']
    for split in splits:
        datasets[split] = textdata(directory=configs['data_dir'], split=split, vocab_file=configs['vocab_file'],
                                   max_sequence_length=configs['max_sequence_length'])
    # prepare FDA drugs dataset
    fda_data = textdata(directory=configs['data_dir'], split=None, vocab_file=configs['vocab_file'],
                        max_sequence_length=configs['max_sequence_length'], filename='all_drugs_only.smi')

    valid_loader = DataLoader(
        dataset=datasets['valid'],
        batch_size=len(datasets['valid'].data),
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    for iteration, valid_batch in enumerate(valid_loader):
        valid_batch_size = valid_batch['inputs'].size(0)
        for k, v in valid_batch.items():
            if torch.is_tensor(v):
                valid_batch[k] = to_cuda_var(v)

    fda_loader = DataLoader(
        dataset=fda_data,
        batch_size=len(fda_data.data),
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    for iteration, fda_batch in enumerate(fda_loader):
        fda_batch_size = fda_batch['inputs'].size(0)
        for k, v in fda_batch.items():
            if torch.is_tensor(v):
                fda_batch[k] = to_cuda_var(v)

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
        unk_idx=datasets['train'].unk_idx,
        prior_var=configs['prior_var']
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
            tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    print(model)

    # define reconstruction loss
    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['wd'])

    # learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # tensorboard tracker
    summary_writer = SummaryWriter(log_dir=root_dir)
    logging.info('Initialization complete. Start training.')

    step = 0
    try:

        if configs ['new_training']:
            start_epoch = 0
            end_epoch = configs['epochs']
        else:
            start_epoch = configs['trained_epochs']
            end_epoch = configs['trained_epochs'] + configs['epochs']

        for epoch in range(start_epoch, end_epoch):

            train_grad_metric = {
                'total_gradient_norm': [],
                'enc_gradient_norm': [],
                'dec_gradient_norm': [],
                'h2mu_gradient_norm': [],
                'h2logv_gradient_norm': []
            }

            for split in splits:

                data_loader = DataLoader(
                    dataset=datasets[split],
                    batch_size=configs['batch_size'],
                    #shuffle=(split=='train'),
                    shuffle=False,
                    num_workers=8,
                    pin_memory=torch.cuda.is_available()
                )

                # get num. of samples in data_loader
                num_samples = len(data_loader.dataset.data.keys())

                epoch_metric = {
                    'total_loss': [],
                    'nll_loss': [],
                    'kl_loss': [],
                    'kl_weight': [],
                    'marginal_posterior_divergence': []
                }

                start_time = time.time()
                for iteration, batch in enumerate(data_loader):
                    # current batch size which can be smaller than default batch size when it is the last batch
                    batch_size = batch['inputs'].size(0)

                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = to_cuda_var(v)

                    # forward
                    logp, mean, logv, z, vt, u = model(batch['inputs'], batch['len'])

                    # loss calculation
                    NLL_loss, KL_loss, KL_weight, marginal_posterior_divergence = loss_fn(configs, NLL, logp, batch['targets'],
                        batch['len'], z, vt, u, mean, logv, step, epoch + 1, start_epoch, num_samples)

                    loss = (NLL_loss + KL_weight * (configs['beta']*KL_loss + configs['alpha']*marginal_posterior_divergence))/batch_size

                    #record metrics
                    epoch_metric['total_loss'].append(loss.item())
                    epoch_metric['nll_loss'].append(NLL_loss.item()/batch_size)
                    epoch_metric['kl_loss'].append(KL_loss.item()/batch_size)
                    epoch_metric['kl_weight'].append(KL_weight)
                    epoch_metric['marginal_posterior_divergence'].append(marginal_posterior_divergence.item()/batch_size)
                    #epoch_metric['lr'].append((optimizer.param_groups[0]['lr']))


                    # backward + optimization
                    if split == 'train':
                        # calculate gradients and update weights
                        optimizer.zero_grad()
                        loss.backward()

                        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
                        torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_norm'])

                        optimizer.step()
                        step += 1

                        # check the norm of weights, norm of gradients
                        total_norm = 0
                        for p in model.parameters():
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_grad_norm = total_norm ** (1. / 2)

                        enc_grad_norm = 0
                        for p in model.encoder_rnn.parameters():
                            param_norm = p.grad.data.norm(2)
                            enc_grad_norm += param_norm.item() ** 2
                        enc_grad_norm = enc_grad_norm ** (1. / 2)

                        dec_grad_norm = 0
                        for p in model.decoder_rnn.parameters():
                            param_norm = p.grad.data.norm(2)
                            dec_grad_norm += param_norm.item() ** 2
                        dec_grad_norm = dec_grad_norm ** (1. / 2)

                        h2m_wght_norm = 0
                        h2m_grad_norm = 0
                        for p in model.hidden2mean.parameters():
                            wght_norm = p.data.norm(2)
                            grad_norm = p.grad.data.norm(2)
                            h2m_wght_norm += wght_norm.item() ** 2
                            h2m_grad_norm += grad_norm.item() ** 2
                        # h2m_wght_norm = h2m_wght_norm ** (1. / 2)
                        h2m_grad_norm = h2m_grad_norm ** (1. / 2)

                        h2v_wght_norm = 0
                        h2v_grad_norm = 0
                        for p in model.hidden2logv.parameters():
                            wght_norm = p.data.norm(2)
                            grad_norm = p.grad.data.norm(2)
                            h2v_wght_norm += wght_norm.item() ** 2
                            h2v_grad_norm += grad_norm.item() ** 2
                        # h2v_wght_norm = h2v_wght_norm ** (1. / 2)
                        h2v_grad_norm = h2v_grad_norm ** (1. / 2)

                        train_grad_metric['total_gradient_norm'].append(total_grad_norm)
                        train_grad_metric['enc_gradient_norm'].append(enc_grad_norm)
                        train_grad_metric['dec_gradient_norm'].append(dec_grad_norm)
                        train_grad_metric['h2mu_gradient_norm'].append(h2m_grad_norm)
                        train_grad_metric['h2logv_gradient_norm'].append(h2v_grad_norm)


                        # logging training status
                        if iteration == 0 or (iteration + 1) % configs['logging_steps'] == 0 or iteration + 1 == len(data_loader):

                            end_time = time.time()

                            # monitor loss during training
                            logging_loss(epoch, end_epoch, iteration, data_loader, loss, NLL_loss, KL_loss, KL_weight, marginal_posterior_divergence, batch_size, start_time, end_time)

                            start_time = time.time()

                        # saving model checkpoint
                        if (epoch + 1) % configs['save_per_epochs'] == 0 and iteration + 1 == len(data_loader):
                            logging.info('Saving model checkpoint...')
                            checkpoint_name = os.path.join(root_dir, 'checkpoint_epoch%03d.model' % (epoch + 1))
                            torch.save(model.state_dict(), checkpoint_name)

                #epoch -> split level (e.g. train, valid, test) + epoch level
                epoch_update_loss(summary_writer, split, epoch, epoch_metric['total_loss'], epoch_metric['nll_loss'], epoch_metric['kl_loss'], epoch_metric['kl_weight'], epoch_metric['marginal_posterior_divergence'])

            # epoch level
            # monitor network gradient during training
            training_grad_norm(epoch, train_grad_metric, summary_writer)

            # fda drugs + epoch level
            mean_f = fda_drugs_evaluation(configs, model, fda_batch, fda_batch_size, NLL, step, epoch, start_epoch, end_epoch, summary_writer)

            # prior: % validity, uniqueness, novelty ~ p(z)
            prior_samples_evaluation(epoch, configs, model, summary_writer)

            # posterior: % accurately reconstructed
            posterior_recon_accuracy(configs, smiles_fda, 'fda', model, summary_writer, epoch)
            posterior_recon_accuracy(configs, smiles_test[:1000], 'test', model, summary_writer, epoch)


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


def posterior_recon_accuracy(configs, smiles_lst, dataset, model, summary_writer, epoch):
    recon_cnt = 0
    for smi in smiles_lst:
        # skip smiles that has characters not in the ZINC char list
        try:
            mu, logv = smiles2mean(configs, smi, model)
            _, _, posterior_smiles_sample = latent2smiles(configs,model, z=mu, nsamples=5, sampling_mode='beam')
            if smi in posterior_smiles_sample:
                recon_cnt+=1
        except:
            pass
    split = 'posterior_samples'
    summary_writer.add_scalar('%s/reconstruction_%s' % (split, dataset), recon_cnt/len(smiles_lst), epoch)


def prior_samples_evaluation(epoch, configs, model, summary_writer):
    # z ~ p(z), sample one smile for each z using 'greedy' sampling method
    samples_idx, z = model.inference(n=1000, sampling_mode='greedy', z=None)
    prior_smiles_sample = idx2smiles(configs, samples_idx)
    perc_valid, perc_chem_valid, perc_unique, perc_novel = eval_prior_samples(configs, prior_smiles_sample)
    split = 'prior_samples'
    summary_writer.add_scalar('%s/validty' % split, perc_valid, epoch)
    summary_writer.add_scalar('%s/uniqueness' % split, perc_unique, epoch)
    summary_writer.add_scalar('%s/novelty' % split, perc_novel, epoch)


def fda_drugs_evaluation(configs, model, fda_batch, fda_batch_size, NLL, step, epoch, start_epoch, end_epoch, summary_writer):

    logp_f, mean_f, logv_f, z_f, vt_f, u_f = model(fda_batch['inputs'], fda_batch['len'])
    NLL_loss_f, KL_loss_f, KL_weight_f, marginal_posterior_divergence_f = loss_fn(configs, NLL, logp_f, fda_batch['targets'],
                                                 fda_batch['len'], z_f, vt_f, u_f, mean_f, logv_f,
                                                 step, epoch + 1, start_epoch, fda_batch_size)
    loss_f = (NLL_loss_f + KL_weight_f * KL_loss_f + marginal_posterior_divergence_f)
    # logging loss per epoch
    logging.info(
        'FDA Drugs: Epoch [{}/{}], Loss: {:.4f}, NLL-Loss: {:.4f}, KL-Loss: {:.4f}, Annealing-Weight: {:.4f}, Marginal Posterior Divergence: {:.4f}'.
            format(epoch + 1, end_epoch, loss_f.item()/fda_batch_size, NLL_loss_f.item() / fda_batch_size,
                   KL_loss_f.item() / fda_batch_size, KL_weight_f, marginal_posterior_divergence_f/fda_batch_size))
    # tensorboard
    split = 'FDA'
    summary_writer.add_scalar('%s/avg_total_loss' % split, loss_f/fda_batch_size, epoch)
    summary_writer.add_scalar('%s/avg_nll_loss' % split, NLL_loss_f/fda_batch_size, epoch)
    summary_writer.add_scalar('%s/avg_kl_loss' % split, KL_loss_f/fda_batch_size, epoch)
    summary_writer.add_scalar('%s/avg_kl_weight' % split, KL_weight_f, epoch)
    summary_writer.add_scalar('%s/avg_marginal_posterior_divergence' % split, marginal_posterior_divergence_f/fda_batch_size, epoch)
    return mean_f


def logging_loss(epoch, end_epoch, iteration, data_loader, loss, NLL_loss, KL_loss, KL_weight, marginal_posterior_divergence, batch_size, start_time, end_time):
    logging.info(
        'Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, NLL-Loss: {:.4f}, KL-Loss: {:.4f}, Annealing-Weight: {:.4f}, Marginal Posterior Divergence: {:.4f}, Speed: {:.4f} ms'.
            format(epoch + 1, end_epoch, iteration + 1, len(data_loader), loss.item(), NLL_loss.item() / batch_size,
                   KL_loss.item() / batch_size, KL_weight, marginal_posterior_divergence/batch_size, (end_time - start_time) * 100 / batch_size))


def training_grad_norm(epoch, train_grad_metric, summary_writer):

    split='train'
    total_grad_norm = np.array(train_grad_metric['total_gradient_norm'])
    enc_grad_norm = np.array(train_grad_metric['enc_gradient_norm'])
    dec_grad_norm = np.array(train_grad_metric['dec_gradient_norm'])
    h2m_grad_norm = np.array(train_grad_metric['h2mu_gradient_norm'])
    h2v_grad_norm = np.array(train_grad_metric['h2logv_gradient_norm'])

    summary_writer.add_scalar('%s/total_gradient_norm_log10' % split, np.log10(total_grad_norm.mean()), epoch)
    summary_writer.add_scalar('%s/encoder_gradient_norm_log10' % split, np.log10(enc_grad_norm.mean()), epoch)
    summary_writer.add_scalar('%s/decoder_gradient_norm_log10' % split, np.log10(dec_grad_norm.mean()), epoch)
    summary_writer.add_scalar('%s/h2mu_graddient_norm_log10' % split, np.log10(h2m_grad_norm.mean()), epoch)
    summary_writer.add_scalar('%s/h2logvar_graddient_norm_log10' % split, np.log10(h2v_grad_norm.mean()), epoch)


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