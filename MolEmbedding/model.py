import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.distributions as dis
from lorentz_model import *
import pandas as pd


class MolVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size,
                 manifold_type,
                 rnn_type, bidirectional, num_layers,
                 word_dropout_rate, embedding_dropout_rate, one_hot_rep,
                 max_sequence_length,
                 sos_idx, eos_idx, pad_idx, unk_idx, prior_var):

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

        self.prior_var = to_cuda_var(torch.tensor(prior_var))
        self.prior_std = self.prior_var.pow(0.5)

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.bidirectional_factor = 2 if bidirectional else 1
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

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        # use a simple decoder (single layer, unidirectional)
        self.decoder_rnn = rnn(embedding_size, hidden_size * self.hidden_factor, num_layers=1, bidirectional=False, batch_first=True)
        # self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        # MLP -> stochastic
        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)

        # stochastic -> MLP
        if manifold_type == 'Euclidean':
            self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        elif manifold_type == 'Lorentz':
            self.latent2hidden = nn.Linear(latent_size + 1, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * self.hidden_factor, vocab_size)

    def encoder(self, batch_size, sorted_lengths, input_sequence):
        # using one-hot rep.
        if self.one_hot_rep:
            input_embedding = self.one_hot_embedding(input_sequence) #[batch_size, max_seq_len-1, one_hot_size]
        # embedding layer
        else:
            input_embedding = self.input_embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input) #[num_layers, batch_size, hidden_size]

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            #hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
            hidden = hidden.permute(1,0,2).reshape(batch_size,self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        return hidden

    def reparameterize(self, batch_size, hidden):
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
            eps = to_cuda_var(torch.randn([batch_size, self.latent_size]))
            z = mean + eps * std
            vt = None
            u = None
        elif self.manifold_type == 'Lorentz':
            vt, u, z = lorentz_sampling(mean, logv)
        return mean, logv, vt, u, z

    def decoder(self, batch_size, input_sequence, sorted_lengths, sorted_idx, z):

        hidden = self.latent2hidden(z)
        hidden = hidden.unsqueeze(0)
        """
        if self.num_layers > 1:
            h = hidden.view(batch_size, self.num_layers, self.hidden_size*self.bidirectional_factor)
            hidden = h.permute(1,0,2)
        else:
            hidden = hidden.unsqueeze(0)
        """

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
        return logp

    def forward(self, input_sequence, sequence_length):

        #batch first = True
        batch_size = input_sequence.size(0) #[bacth_size, max_seq_len-1]
        sorted_lengths, sorted_idx = torch.sort(sequence_length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        #eoncoder
        hidden = self.encoder(batch_size, sorted_lengths, input_sequence)

        #reparameterization
        mean, logv, vt, u, z = self.reparameterize(batch_size, hidden)

        #decoder
        logp = self.decoder(batch_size, input_sequence, sorted_lengths, sorted_idx, z)

        return logp, mean, logv, z, vt, u


    def one_hot_embedding(self, input_sequence):
        embeddings = np.zeros((input_sequence.shape[0], input_sequence.shape[1], self.vocab_size), dtype=np.float32)
        for b, batch in enumerate(input_sequence):
            for t, char in enumerate(batch):
                if char.item() != 0:
                    embeddings[b, t, char.item()] = 1
        return to_cuda_var(torch.from_numpy(embeddings))


    def inference(self, n, sampling_mode, z=None):

        # sampling from prior distribution
        if z is None:
            if self.manifold_type == 'Euclidean':
                batch_size = n
                #z = to_cuda_var(torch.randn(batch_size, self.latent_size))

                prior_mean = to_cuda_var(torch.zeros(self.latent_size))
                prior_cov = to_cuda_var(torch.eye(self.latent_size) * self.prior_var)
                mnd = dis.MultivariateNormal(prior_mean, prior_cov)
                z = mnd.sample([batch_size])

            elif self.manifold_type == 'Lorentz':
                batch_size = n
                mu0 = to_cuda_var(torch.zeros(batch_size, self.latent_size))
                mu0_h = lorentz_mapping_origin(mu0)
                #logvar = to_cuda_var(torch.zeros(self.latent_size).repeat(batch_size, 1))
                logvar = to_cuda_var(torch.ones(self.latent_size).repeat(batch_size, 1) * self.prior_var).log()
                _, _, z = lorentz_sampling(mu0_h, logvar)
        # use the input z
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)
        hidden = hidden.unsqueeze(0)
        """
        if self.bidirectional or self.num_layers > 1:
            #unflatten hidden state
            h = hidden.view(batch_size, self.num_layers, self.hidden_size*self.bidirectional_factor)
            hidden = h.permute(1,0,2)
        else:
            hidden = hidden.unsqueeze(0)
        """

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
                input_embedding = self.input_embedding(input_sequence)

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


    def beam_search(self, z, B):
        #current top B SMILES
        generations_out = self.tensor(B, self.max_sequence_length).fill_(self.pad_idx).long()

        # smile_x = 'CC(C)Cc1ccc(C(C)C(=O)O)cc1'  # Ibuprofen
        zi = to_cuda_var(z.repeat([B, 1]))
        h_in = self.latent2hidden(zi)
        h_in = h_in.unsqueeze(0)
        """
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            #h_in = h_in.view(self.hidden_factor, B, self.hidden_size)
            h = h_in.view(B, self.num_layers, self.hidden_size*self.bidirectional_factor)
            h_in = h.permute(1,0,2)
        else:
            h_in = h_in.unsqueeze(0)
        """
        smiles_eos_lst = []
        t=0
        while(t<self.max_sequence_length):
            if t == 0:
                input_sequence = torch.Tensor(B).fill_(self.sos_idx).long()  # starting with '<sos>'
            input_sequence = input_sequence.unsqueeze(1)
            if self.one_hot_rep:
                input_embedding = self.one_hot_embedding(input_sequence)
            else:
                input_embedding = self.input_embedding(input_sequence)

            output, h_out = self.decoder_rnn(input_embedding, h_in)
            logits = self.outputs2vocab(output)
            logp = F.log_softmax(logits,dim=-1)

            if t==0:
                logp_sequence_out, output_sequence = torch.topk(logp, B, dim=-1)
                logp_sequence_out = logp_sequence_out[0, :, :].squeeze()
                output_sequence = output_sequence[0, :, :].squeeze()
                generations_out[:,t] = output_sequence
            elif t>0:
                logp = logp + logp_sequence_in.view(-1,1,1)
                logp_flat = logp.view(-1)
                logp_sequence_out, flatten_tensor_idx = torch.topk(logp_flat, B)
                tensor_idx = np.divmod(flatten_tensor_idx.cpu(), logp.shape[-1])
                output_sequence = tensor_idx[1]
                h_out = h_out[:, tensor_idx[0].data, :]
                temp = self.tensor(B, self.max_sequence_length).fill_(self.pad_idx).long()
                for g in range(B):
                    b = tensor_idx[0][g]
                    c = tensor_idx[1][g]
                    temp[g,:] = generations_in[b,:]
                    temp[g,t] = c
                generations_out = temp

            #check whether <EOS> is generated in the output_sequence
            running_seqs = torch.arange(0, B, out=self.tensor()).long()
            running_mask = (output_sequence != self.eos_idx).data
            running_idx = running_seqs.masked_select(to_cuda_var(running_mask))

            input_sequence = output_sequence[running_idx]
            h_in = h_out[:,running_idx,:]
            logp_sequence_in = logp_sequence_out[running_idx]
            generations_in = generations_out[running_idx]

            eos_seqs = torch.arange(0, B, out=self.tensor()).long()
            eos_mask = (output_sequence == self.eos_idx).data
            eos_idx = eos_seqs.masked_select(to_cuda_var(eos_mask))
            if eos_idx.sum().item() > 0:
                for j in eos_idx:
                    smiles_eos_lst.append([generations_out[j.item(),:], logp_sequence_out[j.item()].item()])
                smiles_eos_df = pd.DataFrame(smiles_eos_lst, columns=['seqs','logp'])

                if len(running_idx) == 0:
                    break
                # if the minimum logp in smiles_eos_lst is larger than the max of running values and size of smiles_eos_lst >= Beam Width
                stop_flag = (smiles_eos_df.shape[0] >=B) and (smiles_eos_df['logp'].min() >= logp_sequence_in.max())
                if stop_flag:
                    break

            t+=1

        smiles_running_lst = []
        for b in range(len(generations_in)):
            smiles_running_lst.append([generations_in[b,:], logp_sequence_in[b].item()])

        smiles_bs_lst = smiles_eos_lst + smiles_running_lst
        smiles_bs_df = pd.DataFrame(smiles_bs_lst, columns=['seqs', 'logp'])
        df = smiles_bs_df.nlargest(B, 'logp')

        lst = []
        for b in range(B):
            generations = df.iloc[b]['seqs'].cpu().detach().numpy()
            lst.append(generations)
        arrays = np.stack(lst, axis=0)
        generations = torch.from_numpy(arrays)
        return generations, z


    def _sample(self, dist, sampling_mode='greedy'):
        if sampling_mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)

            #debug
            if sample.max().item()>38:
                print(sample.max().item())
                print(dist.shape)

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



