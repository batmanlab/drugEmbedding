import torch.distributions as dis
from drugdata import *
from evae import *
from hvae import *

def load_model(configs):
    dataset = drugdata(task=configs['task'],
                       fda_drugs_dir=configs['data_dir'],
                       fda_smiles_file=configs['fda_file'],
                       fda_vocab_file=configs['vocab_file'],
                       fda_drugs_sp_file=configs['atc_sim_file'],
                       experiment_dir=os.path.join(configs['checkpoint_dir'], configs['experiment_name']),
                       smi_file='smiles_train.smi',
                       max_sequence_length=configs['max_sequence_length'],
                       nneg=configs['nneg']
                       )

    # load model
    if configs['manifold_type'] == 'Euclidean':
        model = EVAE(
            vocab_size=dataset.vocab_size,
            hidden_size=configs['hidden_size'],
            latent_size=configs['latent_size'],
            bidirectional=configs['bidirectional'],
            num_layers=configs['num_layers'],
            word_dropout_rate=configs['word_dropout_rate'],
            max_sequence_length=configs['max_sequence_length'],
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx,
            pad_idx=dataset.pad_idx,
            unk_idx=dataset.unk_idx,
            prior=configs['prior_type'],
            alpha=configs['alpha'],
            beta=configs['beta'],
            gamma=configs['gamma']
        )

    elif configs['manifold_type'] == 'Lorentz':
        model = HVAE(
            vocab_size=dataset.vocab_size,
            hidden_size=configs['hidden_size'],
            latent_size=configs['latent_size'],
            bidirectional=configs['bidirectional'],
            num_layers=configs['num_layers'],
            word_dropout_rate=configs['word_dropout_rate'],
            max_sequence_length=configs['max_sequence_length'],
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx,
            pad_idx=dataset.pad_idx,
            unk_idx=dataset.unk_idx,
            prior=configs['prior_type'],
            alpha=configs['alpha'],
            beta=configs['beta'],
            gamma=configs['gamma']
        )

    del dataset

    torch.no_grad()
    checkpoint_path = os.path.join(configs['checkpoint_dir'] + '/' + configs['experiment_name'], configs['checkpoint'])
    model.load_state_dict(torch.load(checkpoint_path))
    if torch.cuda.is_available():
        model = model.cuda()

    return model

def inference(configs, model, n, sampling_mode, z=None):

    # sampling from prior distribution
    if z is None:
        if configs['manifold_type'] == 'Euclidean':
            batch_size = n
            # z = to_cuda_var(torch.randn(batch_size, self.latent_size))

            prior_mean = to_cuda_var(torch.zeros(configs['latent_size']))
            prior_cov = to_cuda_var(torch.eye(configs['latent_size']))
            mnd = dis.MultivariateNormal(prior_mean, prior_cov)
            z = mnd.sample([batch_size])

        elif configs['manifold_type'] == 'Lorentz':
            batch_size = n
            mu0 = to_cuda_var(torch.zeros(batch_size, configs['latent_size']))
            mu0_h = lorentz_mapping_origin(mu0)
            # logvar = to_cuda_var(torch.zeros(self.latent_size).repeat(batch_size, 1))
            logvar = to_cuda_var(torch.ones(configs['latent_size']).repeat(batch_size, 1)).log()
            _, _, z = lorentz_sampling(mu0_h, logvar)

    # use the input z
    else:
        batch_size = z.size(0)

    hidden = model.latent2hidden(z)
    hidden = hidden.reshape(-1, configs['num_layers'], configs['hidden_size']).permute(1, 0, 2).contiguous()  # hidden_factor, batch_size, hidden_size

    # required for dynamic stopping of sentence generation
    tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    sequence_idx = to_cuda_var(torch.arange(0, batch_size, out=tensor_type()).long())  # all idx of batch
    sequence_running = to_cuda_var(torch.arange(0, batch_size, out=tensor_type()).long())  # all idx of batch which are still generating
    sequence_mask = to_cuda_var(torch.ones(batch_size, out=tensor_type()).byte())

    running_seqs = to_cuda_var(torch.arange(0, batch_size, out=tensor_type()).long())  # idx of still generating sequence with respect to current long

    generations = tensor_type(batch_size, configs['max_sequence_length']).fill_(model.pad_idx).long()

    t=0
    while(t<configs['max_sequence_length'] and len(running_seqs)>0):

        if t==0:
            input_sequence = torch.Tensor(batch_size).fill_(model.sos_idx).long()  # st  rting with '<sos>'

        input_sequence = input_sequence.unsqueeze(1)
        input_embedding = model.one_hot_embedding(input_sequence)

        output, hidden = model.decoder_rnn(input_embedding, hidden)

        logits = model.outputs2vocab(output)

        input_sequence = _sample(logits, sampling_mode)

        if input_sequence.dim() == 0:
            input_sequence = torch.tensor([input_sequence])

        # save next input
        generations = _save_sample(generations, input_sequence, sequence_running, t)

        # update global running sequences
        sequence_mask[sequence_running] = (input_sequence != model.eos_idx).data
        sequence_running = sequence_idx.masked_select(sequence_mask)

        # update local running sequences
        running_mask = (input_sequence != model.eos_idx).data
        running_seqs = running_seqs.masked_select(to_cuda_var(running_mask))

        # prune input and hidden state according to local update
        if len(running_seqs) > 0:
            input_sequence = input_sequence[running_seqs]
            hidden = hidden[:, running_seqs]

            running_seqs = torch.arange(0, len(running_seqs), out=tensor_type()).long()

        t += 1

    return generations, z


def _sample(dist, sampling_mode='greedy'):
    if sampling_mode == 'greedy':
        _, sample = torch.topk(dist, 1, dim=-1)
    elif sampling_mode =='random':
        p = F.softmax(dist, dim=-1)
        sample = dis.Categorical(p).sample()
    sample = sample.squeeze()
    return sample


def _save_sample(save_to, sample, running_seqs, t):
    # select only still running
    running_latest = save_to[running_seqs]
    # update token at position t
    running_latest[:,t] = sample.data
    # save back
    save_to[running_seqs] = running_latest
    return save_to


def smiles_to_tokens(data_dir, vocab_file, s):
    """
    convert a SMILES to a list of tokens
    :param data_dir: directory of vocabulary file
    :param vocab_file: vocabulary file name
    :param s: input SMILES
    :return: a list of tokens in the input SMILES
    """
    with open(os.path.join(data_dir, vocab_file), 'rb') as fp:
        char_list = pickle.load(fp)
    fp.close()

    s = s.strip()

    j = 0
    tokens_lst = []
    while j < len(s):
        # handle atoms with two characters
        if j < len(s) - 1 and s[j:j + 2] in char_list:
            token = s[j:j + 2]
            j = j + 2

        # handle atoms with one character including hydrogen
        elif s[j] in char_list:
            token = s[j]
            j = j + 1

        # handel unknown character
        else:
            token = '<unk>'
            j = j + 1

        tokens_lst.append(token)
    return tokens_lst


def idx2word(directory, vocab_file):
    """
    create two dictionaries: word to index, index to word
    :param directory: directory of vocabulary file
    :param vocab_file: name of vocabulary file
    :return:
    """
    w2i = dict()
    i2w = dict()

    # add special tokens to vocab
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    for st in special_tokens:
        i2w[len(w2i)] = st
        w2i[st] = len(w2i)

    # load unique chars
    with open(os.path.join(directory, vocab_file), 'rb') as fp:
        char_list = pickle.load(fp)
    fp.close()
    for i, c in enumerate(char_list):
        i2w[len(w2i)] = c
        w2i[c] = len(w2i)
    return w2i, i2w


def smiles2mean(configs, smiles_x, model):
    """
    encode a SMILES to the posterior mean of q(z|x)
    :param configs: model configurations
    :param smiles_x: an input SMILES string
    :param model: model checkpoint
    :return: the posterior mean in the latent space (a vector representation)
    """
    w2i, i2w = idx2word(configs['data_dir'], configs['vocab_file'])
    input_sequence = [w2i['<sos>']]
    tokens = smiles_to_tokens(configs['data_dir'], configs['vocab_file'], smiles_x)
    for i in tokens:
        input_sequence.append(w2i[i])
    input_sequence.append(w2i['<eos>'])
    input_sequence = input_sequence + [0] * (configs['max_sequence_length'] - len(input_sequence) - 1)
    input_sequence = np.asarray(input_sequence)
    input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)
    sequence_length = torch.tensor([len(smiles_x) + 1])

    # run through encoder
    hidden = model.encoder(input_sequence, sequence_length)
    if configs['manifold_type'] == 'Euclidean':
        mean, logv, _ = model.reparameterize(hidden)
    elif configs['manifold_type'] == 'Lorentz':
        mean, logv, _, _, _ = model.reparameterize(hidden)
    return mean, logv


def idx2smiles(configs, samples_idx):

    w2i, i2w = idx2word(configs['data_dir'], configs['vocab_file'])
    eos_idx = w2i['<eos>']

    nsamples = samples_idx.shape[0]
    smiles_lst = []
    for i in range(nsamples):
        smiles = []
        for j in range(configs['max_sequence_length']):
            if samples_idx[i,j] == eos_idx:
                break
            smiles.append(i2w[samples_idx[i,j].item()])
        smiles = "".join(smiles)
        smiles_lst.append(smiles)
    return smiles_lst


def latent2smiles(configs, model, z, nsamples, sampling_mode):
    """
    decode a point in latent space to a SMILES string
    :param configs: model configurations
    :param model: model checkpoint
    :param z: input latent point (a vector representation)
    :param nsample: number of generated SMILES (greedy = 1, beam search = beam width, random = nsample)
    :param sampling_mode: greedy, beam search or random (at the output softmax layer)
    :return: z, samples_idx, smiles_lst (input z, one-hot-vector index, a list of sampled SMILES)
    """
    # sample from posterior distribution
    if sampling_mode == 'greedy':
        samples_idx, z = inference(configs, model, nsamples, sampling_mode, z)
    elif sampling_mode == 'random':
        samples_idx, z = inference(configs, model, nsamples, sampling_mode, z)
    smiles_lst = idx2smiles(configs, samples_idx)
    return z, samples_idx, smiles_lst


