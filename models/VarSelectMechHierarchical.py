from asyncio import trsock
import time
import math
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

import models
from utils.util import init_module_weights, gaussian_kld
from utils.util import init_word_embedding
from utils.KL_Vanishing.CyclicalAnnealing import Scaler, frange_cycle_linear
from utils.KL_Vanishing.PID import PIDControl

from .modules.VariationalModules import GaussianVariation, GMMVariation, LGMVariation


from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pad_sequence



"""
VarSelectMech 不要采用 beam search解码，用原始代码的解码器
"""

class attentive_pooling(nn.Module):
    def __init__(self, hidden_size):
        super(attentive_pooling, self).__init__()
        self.w = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, memory, mask=None):
        h = torch.tanh(self.w(memory))
        score = torch.squeeze(self.u(h), -1)
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, 1)
        return s


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirec, encode_rnn_type):
        super(Encoder, self).__init__()
        if num_layers == 1:
            dropout = 0.0

        if encode_rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, bidirectional=bidirec,
                            batch_first=True)
        elif encode_rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, bidirectional=bidirec,
                            batch_first=True)

    def forward(self, input, lengths):
        length, indices = torch.sort(lengths, dim=0, descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = list(torch.unbind(length, dim=0))
        embs = pack(torch.index_select(input, dim=0, index=indices), input_length, batch_first=True)
        outputs, _ = self.rnn(embs)
        outputs = unpack(outputs, batch_first=True)[0]
        outputs = torch.index_select(outputs, dim=0, index=ind)
        return outputs


class VarSelectMechHierarchical(nn.Module):
    def __init__(self, config, vocab, use_cuda=True):
        super(VarSelectMechHierarchical, self).__init__()
        self.config = config
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        self.use_content = self.config.use_content
        self.pad_token_id = vocab._word2id['[PADDING]']

        self.device = config.device

        # Input components
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.config.emb_size,
            padding_idx=self.pad_token_id,
            _weight=init_word_embedding(
                load_pretrained_word_embedding=self.config.use_pretrained_word_embedding,
                pretrained_word_embedding_path=self.config.word_embedding_path,
                id2word=self.vocab._id2word,
                word_embedding_dim=self.config.emb_size,
                vocab_size=self.vocab_size,
                pad_token_id=self.pad_token_id
            ),
        )

        """ 层级注意力机制编码 """
        # 词编码
        self.word_encoder = Encoder(config.emb_size, config.emb_size, config.num_layers, config.dropout, config.bidirec, config.encode_rnn_type)
        self.word_attentive_pool = attentive_pooling(config.emb_size * 2)
        # 句子编码
        self.sentence_encoder = Encoder(config.emb_size * 2, config.emb_size * 2, config.num_layers, config.dropout, config.bidirec, config.encode_rnn_type)
        self.sentence_attentive_pool = attentive_pooling(config.decoder_hidden_size)
        self.w_context = nn.Linear(config.decoder_hidden_size * 2, config.decoder_hidden_size, bias=True)

        # 评论文本编码
        self.encoder = models.rnn_encoder(config, self.vocab_size, embedding=self.embedding)

        self.decoder = models.rnn_decoder(config, self.vocab_size, embedding=self.embedding)
        self.criterion = models.criterion(self.vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax(-1)
        self.tanh = nn.Tanh()

        self.pad_token_id = 0
        is_direct = 2 if self.config.bidirec == True else 1
        self.use_bow_loss = self.config.use_bow_loss if hasattr(self.config, "use_bow_loss") else True
        self.encoder_hidden_size = self.config.encoder_hidden_size * is_direct
        self.latent_variable_dim = self.config.latent_dim

        self.KL_Vanishing = self.config.KL_Vanishing

        # KL_annealing
        self.step = 0
        if self.config.KL_Vanishing == 'KL_annealing':
            self.n_step_annealing = self.config.n_step_annealing if hasattr(config, "n_step_annealing") else 1
        # BN-VAE
        elif self.config.KL_Vanishing == 'BN-VAE':
            self.mu_bn = nn.BatchNorm1d(self.latent_variable_dim, eps=1e-8)
            self.mu_bn.weight.requires_grad = False
            self.mu_bn.weight.fill_(self.config.gamma)
        # Cyclical Annealing Schedule
        elif self.config.KL_Vanishing == 'CyclicalAnnealing':
            self.train_sample_num = self.config.train_sample_num
            total_update = math.ceil(self.train_sample_num/self.config.batch_size) * self.config.epoch
            self.cyc_coef = frange_cycle_linear(0.0, 1.0, total_update, 4)
        # ControlVAE: Controllable Variational Autoencoder
        elif self.config.KL_Vanishing == 'ControlVAE':
            self.pid = PIDControl()
            self.Kp = 0.01
            self.Ki = -0.0001
            self.exp_kl = 2.0

        # Variational components
        if self.config.n_components == 1:
            self.prior_net = GaussianVariation(
                input_dim=self.encoder_hidden_size,
                z_dim=self.latent_variable_dim,
                # large_mlp=True
            )
        elif self.config.n_components > 1:
            if self.config.gaussian_mix_type == "gmm":
                self.prior_net = GMMVariation(
                    input_dim=self.encoder_hidden_size,
                    z_dim=self.latent_variable_dim,
                    n_components=self.config.n_components,
                )
            elif self.config.gaussian_mix_type == "lgm":
                self.prior_net = LGMVariation(
                    input_dim=self.encoder_hidden_size,
                    z_dim=self.latent_variable_dim,
                    n_components=self.config.n_components,
                )

        self.post_net = GaussianVariation(
            input_dim=self.encoder_hidden_size + self.encoder_hidden_size,
            z_dim=self.latent_variable_dim,
        )

        self.latent_to_bow = nn.Sequential(
            nn.Linear(
                self.latent_variable_dim + self.encoder_hidden_size,
                self.latent_variable_dim
            ),
            nn.Tanh(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.latent_variable_dim,
                self.config.vocab_size
            )
        )

        self.ctx_fc = nn.Sequential(
            nn.Linear(
                self.latent_variable_dim + self.encoder_hidden_size,
                self.encoder_hidden_size,
            ),
            nn.Tanh(),
            nn.Dropout(self.config.dropout)
        )

        # the combine mode between variant and selectmech, mdoel = 'serial', 'parallel'
        self.mode = self.config.mode

        # add the select mechanism
        self.num_mappings = self.config.num_mappings
        self.decoder_hidden_size = self.config.decoder_hidden_size
        self.criterion = models.criterion(self.vocab_size, use_cuda)
        self.tau = self.config.tau

        # define self.mappings function
        # self.mappings = nn.ModuleList([nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size) for i in range(self.num_mappings)])
        self.mappings = nn.ModuleList([])
        for i in range(self.num_mappings):
            self.mappig_fc = nn.Sequential(
                nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
                nn.Tanh(),
                nn.Dropout(self.config.dropout)
            )
            self.mappings.append(self.mappig_fc)


        if self.config.mode == 'parallel':
            self.hidden_fc = nn.Sequential(
                nn.Linear(
                    self.decoder_hidden_size * 2,
                    self.decoder_hidden_size,
                ),
                nn.Tanh(),
                nn.Dropout(self.config.dropout)
            )

        # Initialization
        self._init_weights()

    def _init_weights(self,):
        init_module_weights(self.latent_to_bow)
        init_module_weights(self.ctx_fc)
        init_module_weights(self.w_context)

    def hidden_type_squeeze(self, state):
        if self.config.rnn_type == 'lstm':
            hiddens = state[0]
            Cs = state[1]
            if self.config.num_layers == 1:
                hidden = hiddens.squeeze(0)
                C = Cs.squeeze(0)
            else:
                hidden = hiddens[1, :, :].squeeze(0)
                C = Cs[1, :, :].squeeze(0)
        elif self.config.rnn_type == 'gru':
            hiddens = state
            if self.config.num_layers == 1:
                hidden = hiddens.squeeze(0)
            else:
                hidden = hiddens[1, :, :].squeeze(0)

        return hidden

    def hidden_type_unsqueeze(self, hidden):
        if self.config.rnn_type == 'lstm':
            if self.config.num_layers == 1:
                hidden = hidden.unsqueeze(0)
                C = C.unsqueeze(0)
                hiddens = (hidden, C)
            else:
                # hiddens[0, :, :]，存在维度问题，因此将其hidden进行复制
                hiddens = hidden.unsqueeze(0).repeat(self.config.num_layers, 1,  1)
                Cs = C.unsqueeze(0).repeat(self.config.num_layers, 1,  1)
                hiddens = (hiddens, Cs)
        elif self.config.rnn_type == 'gru':
            if self.config.num_layers == 1:
                hiddens = hidden.unsqueeze(0)
            else:
                hiddens = hidden.unsqueeze(0).repeat(self.config.num_layers, 1,  1)

        return hiddens

    def encode(self, src, src_len):
        """
            src         torch.Size([16, 19])
            src_len     torch.Size([16])
            src_mask    torch.Size([16, 19])
        """
        contexts, state = self.encoder(src, src_len)
        return contexts, state

    def article_encode(self, contents, contents_mask, contents_length, sent_mask):
        """ article encoding with attention """
        sent_vec_batch = []

        for content, content_mask, content_length in zip(contents, contents_mask, contents_length):
            length = torch.sum(content_mask, -1)
            emb = self.embedding(content)                                       # emb = torch.Size([10, 100, 256])
            context = self.word_encoder(emb, length)                            # context = torch.Size([seq_num, seq_len, hidden]) ---> torch.Size([10, 100, 512])
            sent_vec = self.word_attentive_pool(context, content_mask)          # torch.Size([10, 512])
            sent_vec_batch.append(sent_vec)
            assert len(sent_vec) == content_length, (len(sent_vec), content_length)  # sentence number

        sent_vec_batch = pad_sequence(sent_vec_batch, batch_first=True)         # torch.Size([32, 45, 512])
        sent_hidden = self.sentence_encoder(sent_vec_batch, contents_length)    # torch.Size([B, 45, hidden_dim])
        sent_hidden = self.w_context(sent_hidden)
        state = self.sentence_attentive_pool(sent_hidden, sent_mask)

        return sent_hidden, state

    def _get_prior_z(self, prior_net_input, assign_k=None, return_pi=False):
        if self.config.n_components == 1:
            ret = self.prior_net(
                                context=prior_net_input,
                            )
        else:
            if self.config.gaussian_mix_type == "gmm":
                ret = self.prior_net(
                                    context=prior_net_input,
                                    gumbel_temp = 0.1
                                )
            elif self.config.gaussian_mix_type == "lgm":
                ret = self.prior_net(
                                    context=prior_net_input,
                                    assign_k=assign_k,
                                    return_pi=return_pi
                                )

        return ret

    def _get_post_z(self, post_net_input):
        z, mu, var = self.post_net(post_net_input)
        return z, mu, var

    def _get_ctx_for_decoder(self, z, dial_encodings):
        ctx_encodings = self.ctx_fc(torch.cat([z, dial_encodings], dim=1))
        # ctx_encodings = self.ctx_fc(z)
        return ctx_encodings

    def _combine_hidden(self, var_hidden, slctmech_hidden):
        ctx_encodings = self.hidden_fc(torch.cat([var_hidden, slctmech_hidden], dim=1))
        return ctx_encodings

    # Resolve posterior collapse
    def _annealing_coef_term(self, step):
        return min(1.0, 1.0*step/self.n_step_annealing)

    def _get_attn_mask(self, attn_keys):
        attn_mask = (attn_keys != self.pad_token_id)
        return attn_mask

    def _ranking(self, article_hidden, hidden):
        """
        Reranking generated responses.
        """
        scores = []

        # 每个 hidden 在通道数量的值
        for index in range(self.num_mappings):
            pred_hidden = self.mappings[index](hidden)

            score = torch.bmm(torch.unsqueeze(pred_hidden, dim=1),
                              torch.unsqueeze(article_hidden, dim=2)).squeeze()
            scores.append(score)

        # 叠加每个 score 的值
        scores = torch.stack(scores, dim=-1)                # all mapping value

        if self.config.ranking_type == 'top-1':
            score_indexs = torch.argmax(scores, dim=-1)
        elif self.config.ranking_type == 'top-beam_size':
            topk_scores = torch.topk(scores, self.config.beam_size)
            # 取最大的top-k值
            score_indexs = topk_scores[1][:self.config.beam_size]
        else:
            raise ValueError(" ranking_type {self.config.ranking_type} not exist!!! ")
        return score_indexs

    def gumbel_softmax(self, logits, tau, eps=1e-10):
        u = torch.rand_like(logits)
        u.requires_grad = False
        gumbel = 0.0 - torch.log(eps - torch.log(u + eps))
        y = logits + gumbel
        return F.softmax(y / tau, dim=-1)

    def collect_metrics(self, outputs):
        metrics = {}

        # Matching Loss
        pos_logits = outputs["pos_logits"]
        # 创建标签pos=1标签
        pos_label = torch.ones_like(pos_logits).fill_(1).to(torch.float32)
        pos_label.requires_grad = False

        # 创建标签neg=0标签
        neg_logits = outputs["neg_logits"]
        neg_label = torch.zeros_like(neg_logits).fill_(0).to(torch.float32)
        neg_label.requires_grad = False

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_label, reduction='mean')
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_label,  reduction='mean')

        match = torch.mean(pos_loss + neg_loss)

        pos_acc = torch.mean(torch.lt(neg_label, pos_logits).to(torch.float32))
        neg_acc = torch.mean(torch.lt(neg_logits, neg_label).to(torch.float32))
        match_acc = (pos_acc + neg_acc) / 2.0

        metrics.update({"match_loss": match, "match_acc": match_acc})

        return metrics

    def select_mechanism(self, post_hidden, post_comment_hidden, is_training=True):
        """
        post_hidden: article encode hidden info
        post_comment_hidden: comment encode hidden info
        """
        outputs = {}

        # Multi-Mapping candidate_hiddens: (batch_size, num_mappings, hidden_dim)
        candidate_hiddens = torch.stack([mapping(post_hidden) for mapping in self.mappings], dim=1)

        # encode the comment
        response_hidden = post_comment_hidden

        # For simplicity, use the target responses in the same batch as negative examples
        neg_response_hidden = torch.flip(response_hidden, dims=[0])

        # (1) 计算 pos_logits 和 neg_logits的方式一
        pos_logits = torch.sum(post_hidden * response_hidden, dim=1, keepdim=True)      # post_hidden and response_hidden shape is the same.
        neg_logits = torch.sum(post_hidden * neg_response_hidden, dim=1, keepdim=True)
        outputs.update({"pos_logits": pos_logits,
                        "neg_logits": neg_logits})

        # Posterior Mapping Selection --> Sampling Probability, similarity=(batch_size, num_mappings)
        similarity = torch.squeeze(torch.bmm(candidate_hiddens, torch.unsqueeze(response_hidden, dim=2)), dim=2)
        post_probs = F.softmax(similarity, dim=-1)

        outputs.update({"post_probs": post_probs})

        # Sample mapping id z
        if is_training:
            z = self.gumbel_softmax(torch.log(post_probs + 1e-10), tau=self.tau)

            # # 训练时，z是soft one-hot的形式(通过softmax加温度来近似)
            # z = F.gumbel_softmax(logits=post_probs, tau=self.tau)
        else:
            indices = torch.argmax(post_probs, dim=1)
            z = F.one_hot(torch.reshape(indices, shape=(-1, 1)), num_classes=self.num_mappings).squeeze(1)
        z = z.float()       # z = [batch, num_mappings]

        # dec_hidden: (batch_size, hidden_size)
        dec_hidden = torch.squeeze(torch.bmm(torch.unsqueeze(z, dim=1), candidate_hiddens), dim=1)

        return outputs, dec_hidden

    def forward(self, batch, use_cuda):
        """Input:
                batch: a package of data
                use_cuda: is use the cuda

            Output:
                final rnn output
        """
        # ---------------- Input data ----------------
        src, src_mask, src_len, sent_mask = batch.sentence_content, batch.sentence_content_mask, batch.sentence_content_len, batch.sentence_mask
        tgt, tgt_len, tgt_mask = batch.tgt, batch.tgt_len, batch.tgt_mask

        if use_cuda:
            src = [s.to(self.device) for s in src]
            src_mask = [s.to(self.device) for s in src_mask]
            src_len, sent_mask = src_len.to(self.device), sent_mask.to(self.device)
            tgt, tgt_len = tgt.to(self.device), tgt_len.to(self.device)

        Y_in = tgt[:, :-1].contiguous()
        Y_out = tgt[:, 1:].contiguous()

        self.Y_out_train = Y_out


        # --------------- Encoding --------------------
        # encoding the input article
        contexts, hidden = self.article_encode(src, src_mask, src_len, sent_mask)   # hidden = torch.Size([32, 256])

        prior_net_input = hidden                                                    # prior_net_input = torch.Size([32, 500])
        # Get prior z
        prior_z, prior_mu, prior_var = self._get_prior_z(prior_net_input)           # prior_z torch.Size([32, 200]), prior_mu torch.Size([32, 200]), prior_var torch.Size([32, 200])

        # BN-VAE
        if self.KL_Vanishing == 'BN-VAE':
            prior_mu = self.mu_bn(prior_mu)

        # Get post z
        # encode the comment
        _, post_comment_encodings = self.encode(tgt, tgt_len)                          # post_sent_encodings=torch.Size([1, 64, 256])
        post_comment_encodings = self.hidden_type_squeeze(post_comment_encodings)            # post_sent_encodings=torch.Size([64, 256])

        post_net_input = torch.cat([post_comment_encodings, hidden], dim=1)            # post_net_input=torch.Size([32, 1000])
        post_z, post_mu, post_var = self._get_post_z(post_net_input)                # post_z torch.Size([32, 200]) post_mu torch.Size([32, 200]), post_var torch.Size([32, 200])

        # self variable for computing loss
        self.post_mu = post_mu
        self.post_var = post_var
        self.prior_mu = prior_mu
        self.prior_var = prior_var
        self.post_z = post_z
        self.hidden = hidden

        # Decode
        var_hidden = self._get_ctx_for_decoder(post_z, hidden)                   # ctx_encodings = torch.Size([32, 500])

        # add the select mechanism
        # hidden from the VAE; comment hidden from post_comment_encodings
        if self.mode == 'serial':
            outputs_slctmech, dec_hidden = self.select_mechanism(var_hidden, post_comment_encodings, is_training=True)
        else:
            outputs_slctmech, slctmech_hidden = self.select_mechanism(prior_net_input, post_comment_encodings, is_training=True)
            # combine the variant  hidden and selectmech hidden
            dec_hidden = self._combine_hidden(var_hidden, slctmech_hidden)


        hiddens = self.hidden_type_unsqueeze(dec_hidden)

        if self.config.teacher_forcing_ratio == 1.0:
            outputs, _, _ = self.decoder(inputs=Y_in, init_state=hiddens, contexts=contexts)                      # only teacher enforce
        else:
            outputs, _, _ = self.decoder.random_teacher_enforce(Y_in,                                             # randome adjusr teacher enforce or generation token
                                                                hiddens,
                                                                contexts,
                                                                teacher_forcing_ratio=self.config.teacher_forcing_ratio)
        # outputs [B, seq_len, vocabs_dim]
        data = {'outputs': outputs,
                'outputs_slctmech': outputs_slctmech,
        }

        return data

    def evaluate(self, batch, use_cuda, assign_k=None):
        src, src_mask, src_len, sent_mask = batch.sentence_content, batch.sentence_content_mask, batch.sentence_content_len, batch.sentence_mask
        tgt, tgt_len, tgt_mask = batch.tgt, batch.tgt_len, batch.tgt_mask

        if use_cuda:
            src = [s.to(self.device) for s in src]
            src_mask = [s.to(self.device) for s in src_mask]
            src_len, sent_mask = src_len.to(self.device), sent_mask.to(self.device)
            tgt, tgt_len = tgt.to(self.device), tgt_len.to(self.device)

        Y_in = tgt[:, :-1].contiguous()
        Y_out = tgt[:, 1:].contiguous()

        self.Y_out_eval = Y_out

        with torch.no_grad():
            contexts, hidden = self.article_encode(src, src_mask, src_len, sent_mask)   # hidden = torch.Size([32, 256])

            prior_net_input = hidden
            prior_z, prior_mu, prior_var = self._get_prior_z(prior_net_input=prior_net_input, assign_k=assign_k )

            # Get post z from comment
            _, post_comment_encodings = self.encode(tgt, tgt_len)                          # post_sent_encodings=torch.Size([1, 64, 256])
            post_comment_encodings = self.hidden_type_squeeze(post_comment_encodings)          # extract hiiden layer
            post_net_input = torch.cat([post_comment_encodings, hidden], dim=1)
            post_z, post_mu, post_var = self._get_post_z(post_net_input)

            self.post_z_eval = post_z
            self.post_mu_eval = post_mu
            self.post_var_eval  = post_var
            self.prior_mu_eval  = prior_mu
            self.prior_var_eval  = prior_var
            self.hidden = hidden

            # Decode from post z
            var_post_hidden = self._get_ctx_for_decoder(post_z, hidden)

            # add the select mechanism
            if self.mode == 'serial':
                outputs_slctmech_post, dec_hidden = self.select_mechanism(var_post_hidden, post_comment_encodings, is_training=False)
            else:
                outputs_slctmech_post, dec_hidden = self.select_mechanism(prior_net_input, post_comment_encodings, is_training=False)
                # combine the variant  hidden and selectmech hidden
                dec_hidden = self._combine_hidden(var_post_hidden, dec_hidden)


            hiddens = self.hidden_type_unsqueeze(dec_hidden)

            if self.config.teacher_forcing_ratio == 1.0:
                post_outputs, _, _ = self.decoder(inputs=Y_in, init_state=hiddens, contexts=contexts)                      # only teacher enforce
            else:
                post_outputs, _, _ = self.decoder.random_teacher_enforce(Y_in,                                             # randome adjusr teacher enforce or generation token
                                                                        hiddens,
                                                                        contexts,
                                                                        teacher_forcing_ratio=self.config.teacher_forcing_ratio
                                                                        )


            # Decode from prior z
            var_prior_hidden = self._get_ctx_for_decoder(prior_z, hidden)

            # add the select mechanism
            if self.mode == 'serial':
                outputs_slctmech_prior, dec_hidden = self.select_mechanism(var_prior_hidden, post_comment_encodings, is_training=False)
            else:
                outputs_slctmech_prior, dec_hidden = self.select_mechanism(prior_net_input, post_comment_encodings, is_training=False)
                # combine the variant  hidden and selectmech hidden
                dec_hidden = self._combine_hidden(var_prior_hidden, dec_hidden)

            hiddens = self.hidden_type_unsqueeze(dec_hidden)

            if self.config.teacher_forcing_ratio == 1.0:
                prior_outputs, _, _ = self.decoder(inputs=Y_in, init_state=hiddens, contexts=contexts)                      # only teacher enforce
            else:
                prior_outputs, _, _ = self.decoder.random_teacher_enforce(Y_in,                                             # randome adjusr teacher enforce or generation token
                                                                        hiddens,
                                                                        contexts,
                                                                        teacher_forcing_ratio=self.config.teacher_forcing_ratio
                                                                        )

        data = {'post_outputs': post_outputs,
                'outputs_slctmech_post': outputs_slctmech_post,
                'prior_outputs': prior_outputs,
                'outputs_slctmech_prior': outputs_slctmech_prior,
                }

        return data

    def compute_loss(self, outputs, targets):                       # targets = torch.Size([51, 64])
        loss = 0.0

        # calc the select mechanism
        outputs_slctmech = outputs['outputs_slctmech']
        outputs_slctmech_loss = self.collect_metrics(outputs_slctmech)
        match_loss = outputs_slctmech_loss['match_loss']
        match_acc = outputs_slctmech_loss['match_acc']

        loss += match_loss

        # calc the variational model loss
        outputs = outputs['outputs'].transpose(0, 1)                # outputs = torch.Size([51, 64, 30000])
        batch_size = targets.size(1)

        sent_loss, acc = models.cross_entropy_loss(outputs, targets, self.criterion)
        loss += sent_loss

        ppl = torch.exp(sent_loss)

        # KLD Loss
        kld_losses = gaussian_kld(self.post_mu,
                                  self.post_var,
                                  self.prior_mu,
                                  self.prior_var,)
        avg_kld = kld_losses.mean()

        # Modifying the value of KL by coefficients(kld_coef)
        if self.KL_Vanishing == 'KL_annealing':
            kld_coef = self._annealing_coef_term(self.step)
        elif self.KL_Vanishing == 'CyclicalAnnealing':
            kld_coef = self.cyc_coef[self.step]
        elif self.config.KL_Vanishing == 'ControlVAE':
            kld_coef = self.pid.pid(self.exp_kl, avg_kld.item(), self.Kp, self.Ki)   # Not involved in the calculation of the neural graph
        else:
            # 默认的系数为1
            kld_coef = 1

        loss += avg_kld*kld_coef

        # BOW
        if self.use_bow_loss:
            Y_out_mask = (self.Y_out_train != self.pad_token_id).float()
            bow_input = torch.cat([self.post_z, self.hidden], dim=1)
            bow_logits = self.latent_to_bow(bow_input)  # [batch_size, vocab_size]
            bow_loss = -F.log_softmax(bow_logits, dim=1).gather(1, self.Y_out_train) * Y_out_mask
            bow_loss = bow_loss.sum(1).mean()
            loss += bow_loss

        data = {
            "loss": loss,
            "ppl": ppl,
            "acc": acc,
            "kld_term": torch.tensor([kld_coef]).to(self.device),
            "kld": avg_kld,
            "prior_abs_mu_mean": self.prior_mu.abs().mean(),
            "prior_var_mean": self.prior_var.mean(),
            "post_abs_mu_mean": self.post_mu.abs().mean(),
            "post_var_mean": self.post_var.mean(),
            "match_loss": match_loss,
            "match_acc": match_acc

        }

        if self.use_bow_loss:
            data["bow_loss"] = bow_loss

        return data

    def eval_computing_loss(self, outputs, targets):
        # calc select mechanism
        outputs_slctmech_post = outputs['outputs_slctmech_post']
        outputs_slctmech_prior = outputs['outputs_slctmech_prior']

        # calc psot loss
        slctmech_post_loss = self.collect_metrics(outputs_slctmech_post)
        match_loss_psot = slctmech_post_loss['match_loss']
        match_acc_psot = slctmech_post_loss['match_acc']

        # calc prior loss
        slctmech_prior_loss = self.collect_metrics(outputs_slctmech_prior)
        match_loss_prior = slctmech_prior_loss['match_loss']
        match_acc_prior = slctmech_prior_loss['match_acc']

        match_loss = (match_loss_psot + match_loss_prior) / 2

        # calc the variational model loss
        post_outputs = outputs['post_outputs'].transpose(0, 1)                              # hidden_outputs [B, seq_len, vocabs_dim], [80, 51, 30000]
        prior_outputs = outputs['prior_outputs'].transpose(0, 1)

        targets = self.Y_out_eval.transpose(0, 1)

        # Reconstruction
        post_word_losses, post_acc = models.cross_entropy_loss(post_outputs, targets, self.criterion)
        post_sent_loss = post_word_losses
        post_ppl = torch.exp(post_sent_loss)

        # Generation
        prior_word_losses, _ = models.cross_entropy_loss(prior_outputs, targets, self.criterion)
        prior_ppl = torch.exp(prior_word_losses)

        # KLD
        kld_losses = gaussian_kld(self.post_mu_eval,
                                  self.post_var_eval,
                                  self.prior_mu_eval,
                                  self.prior_var_eval)
        avg_kld = kld_losses.mean()

        # monitor
        loss = post_sent_loss + avg_kld + match_loss

        data = {
            'ppl': post_ppl,
            "post_ppl": post_ppl,
            "prior_ppl": prior_ppl,
            "loss": loss,
            "acc": post_acc,
            "kld": avg_kld,
            "post_abs_mu_mean": self.post_mu_eval.abs().mean(),
            "post_var_mean": self.post_var_eval.mean(),
            "prior_abs_mu_mean": self.prior_mu_eval.abs().mean(),
            "prior_var_mean": self.prior_var_eval.mean(),
            "match_loss_psot": match_loss_psot,
            "match_acc_psot": match_acc_psot,
            "match_loss_prior": match_loss_prior,
            "match_acc_prior": match_acc_prior,
        }

        return data

    def sample(self, batch, use_cuda, assign_k=None):
        """ generation sampling """
        src, src_mask, src_len, sent_mask = batch.sentence_content, batch.sentence_content_mask, batch.sentence_content_len, batch.sentence_mask

        if use_cuda:
            src = [s.to(self.device) for s in src]
            src_mask = [s.to(self.device) for s in src_mask]
            src_len, sent_mask = src_len.to(self.device), sent_mask.to(self.device)

        with torch.no_grad():
            # encoding the article info
            contexts, article_hidden = self.article_encode(src, src_mask, src_len, sent_mask)   # hidden = torch.Size([32, 256])

            batch_size = contexts.size(0)
            prior_net_input = article_hidden

            if self.config.gaussian_mix_type == "lgm":
                prior_z, prior_mu, prior_var, prior_pi = self._get_prior_z(
                    prior_net_input=prior_net_input,
                    assign_k=assign_k,
                    return_pi=True
                )
            else:
                prior_z, prior_mu, prior_var = self._get_prior_z(
                    prior_net_input=prior_net_input,
                    assign_k=assign_k,
                    return_pi=True
                )

            hidden = self._get_ctx_for_decoder(prior_z, article_hidden)     # hidden = [batch, hidden_size]

            if batch_size != 1:
                batch_size = 1

                hidden = hidden[0, :].unsqueeze(0).contiguous()
                contexts = contexts[0, :, :].unsqueeze(0).contiguous()
                article_hidden = article_hidden[0, :].unsqueeze(0).contiguous()


            """ 解码部分本实验中分为两种：greedy 和 beam
                且在解码时分为了两种机制： 一种是排序算法； 另外一种是选择机制算法；
                我们论文采用了选择机制解码效果更好一点，将最优选择器获得内容作为 decoder 的init hidden
             """
            if self.config.gen_type == 'greedy':
                # add the select mechansim
                if self.config.ranking_type == 'random':

                    if self.config.decoding_select_mechanism == False:
                        # Solve the main.py seting random seed
                        timestamp = time.time()
                        timestamp = int(round(timestamp * 1000000))
                        random.seed(timestamp)

                        # random choose a mapping from self.mappings
                        specific_mapping = random.randint(0, self.num_mappings-1)
                        dec_hidden = self.mappings[specific_mapping](hidden)       # torch.Size([B, hidden_size])

                    elif self.config.decoding_select_mechanism == True:
                        # 采用选择机制去计算
                        outputs, dec_hidden = self.select_mechanism(hidden, article_hidden)

                elif self.config.ranking_type == 'top-1':
                    if self.config.decoding_select_mechanism == False:
                        # 1.ranking algorithm,采用单独的排序算法计算decoder hidden
                        max_score_index = self._ranking(article_hidden, hidden)
                        dec_hidden = self.mappings[max_score_index](hidden)

                    elif self.config.decoding_select_mechanism == True:
                        # 2.采用选择机制去计算
                        outputs, dec_hidden = self.select_mechanism(hidden, article_hidden)

                hiddens = self.hidden_type_unsqueeze(dec_hidden)
                bos = torch.ones(batch_size).long().to(self.device)
                sample_ids, _ = self.decoder.sample(
                                                    input=[bos],
                                                    init_state=hiddens,
                                                    contexts=contexts,
                                                    )

                return sample_ids

            elif self.config.gen_type == 'beam':
                sample_ids_multi = []

                # 确定迭代的次数
                if self.config.ranking_type == 'top-beam_size':
                    num_iteration = self.config.beam_size
                else:
                    num_iteration = self.num_mappings


                for index in range(num_iteration):
                    if self.config.ranking_type == 'top-beam_size':

                        if self.config.decoding_select_mechanism == False:
                            top_beam_size_indexs = self._ranking(article_hidden, hidden)
                            top_beam_size_index = top_beam_size_indexs[index]
                            dec_hidden = self.mappings[top_beam_size_index](hidden)

                        elif self.config.decoding_select_mechanism == True:
                            # 采用选择机制去计算
                            outputs, dec_hidden = self.select_mechanism(hidden, article_hidden)

                    else:
                        if self.config.decoding_select_mechanism == False:
                            dec_hidden = self.mappings[index](hidden)       # torch.Size([B, hidden_size])

                        elif self.config.decoding_select_mechanism == True:
                            # 采用选择机制去计算
                            outputs, dec_hidden = self.select_mechanism(hidden, article_hidden)

                    hiddens = self.hidden_type_unsqueeze(dec_hidden)
                    bos = torch.ones(batch_size).long().to(self.device)
                    sample_ids, _ = self.decoder.sample(
                                                        input=[bos],
                                                        init_state=hiddens,
                                                        contexts=contexts,
                                                        )
                    sample_ids_multi.append(sample_ids)

                sample_ids_multi = torch.cat(sample_ids_multi, dim=0)
                return sample_ids_multi
