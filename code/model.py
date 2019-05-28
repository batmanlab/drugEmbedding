import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.distributions as dis
from utils import *
from lorentz_model import *



class MolVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size,
                 manifold_type,
                 rnn_type, bidirectional, num_layers,
                 word_dropout_rate, embedding_dropout_rate, one_hot_rep,
                 max_sequence_length,
                 sos_idx, eos_idx, pad_idx, unk_idx):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.manifold_type = manifold_type

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.word_dropout_rate = word_dropout_rate
        self.embedding_dropout_rate = embedding_dropout_rate
        self.one_hot_rep = one_hot_rep

        #define embedding layer
        if not self.one_hot_rep:
            self.input_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.pad_idx)
            self.input_embedding_dropout = nn.Dropout(p=self.embedding_dropout_rate)

        #define
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        # MLP -> stochastic
        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        #self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, 1) # try isotropic normal distribution

        # stochastic -> MLP
        if manifold_type == 'Euclidean':
            self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        elif manifold_type == 'Lorentz':
            self.latent2hidden = nn.Linear(latent_size + 1, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)


    def forward(self, input_sequence, sequence_length):

        #batch first = True
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(sequence_length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        #encoder
        # using one-hot rep.
        if self.one_hot_rep:
            input_embedding = self.one_hot_embedding(input_sequence)
        # embedding layer
        else:
            input_embedding = self.input_embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # stochastic layer and reparameterization
        if len(hidden.shape) == 1:
            hidden = hidden.view(1, -1)  # if batch size = 1
        if self.manifold_type == 'Euclidean':
            mean = self.hidden2mean(hidden)
        elif self.manifold_type == 'Lorentz':
            mean = lorentz_mapping(self.hidden2mean(hidden))
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        if self.manifold_type == 'Euclidean':
            z = to_cuda_var(torch.randn([batch_size, self.latent_size]))
            z = z * std + mean
        elif self.manifold_type == 'Lorentz':
            vt, u, z = lorentz_sampling(mean, logv)

        #decoder
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        #decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            if self.one_hot_rep:
                input_embedding = self.one_hot_embedding(decoder_input_sequence)
            # embedding layer
            else:
                input_embedding = self.input_embedding(decoder_input_sequence)
            packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = F.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.vocab_size)


        return logp, mean, logv, z


    def one_hot_embedding(self, input_sequence):
        embeddings = np.zeros((input_sequence.shape[0], input_sequence.shape[1], self.vocab_size), dtype=np.float32)
        for b, batch in enumerate(input_sequence):
            for t, char in enumerate(batch):
                if char.item() != 0:
                    embeddings[b, t, char.item()] = 1
        return to_cuda_var(torch.from_numpy(embeddings))


    def inference(self, n, z=None, sampling_mode = 'greedy'):

        if z is None:
            if self.manifold_type == 'Euclidean':
                batch_size = n
                z = to_cuda_var(torch.randn(batch_size, self.latent_size))
            elif self.manifold_type == 'Lorentz':
                batch_size = n
                mu0 = to_cuda_var(torch.zeros(batch_size, self.latent_size))
                mu0_h = lorentz_mapping_origin(mu0)
                logvar = to_cuda_var(torch.zeros(self.latent_size).repeat(batch_size, 1))
                _, _, z = lorentz_sampling(mu0_h, logvar)
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            #unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        #required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() #all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() #all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() #idx of still generating sequence with respect to current long

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t==0:
                input_sequence = torch.Tensor(batch_size).fill_(self.sos_idx).long() #starting with '<sos>'

            input_sequence = input_sequence.unsqueeze(1)

            if self.one_hot_rep:
                input_embedding = self.one_hot_embedding(input_sequence)
            else:
                input_embedding = self.input_embedding(input_sequence) #TODO GPU bug

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits, sampling_mode)

            if input_sequence.dim() == 0:
                input_sequence = torch.tensor([input_sequence])

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            #update global running sequences
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            #update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(to_cuda_var(running_mask))

            #prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z



    def _sample(self, dist, sampling_mode='greedy'):
        # TODO add beam search
        if sampling_mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)

        elif sampling_mode =='random':
            p = F.softmax(dist, dim=-1)
            sample = dis.Categorical(p).sample()

        sample = sample.squeeze()
        return sample


    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to



