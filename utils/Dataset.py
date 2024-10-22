import json
import random
from collections import OrderedDict
import math
import copy
from statistics import mode
import os

import numpy as np
import torch

from utils.util import split_chinese_sentence


PAD = 0
BOS = 1
EOS = 2
UNK = 3
MASK = 4
TITLE = 5
BUFSIZE = 4096000

MAX_ARTICLE_LENGTH = 600
MAX_TITLE_LENGTH = 30
MAX_COMMENT_LENGTH = 50
MAX_LENGTH = 100


# MAX_COMMENT_NUM = 5
HierarchicalModels = ['hierarchical_attention', 'LongDocuments', 'LongDocuments_v2', 'LongDocuments_v3', 'VarSelectMechHierarchical',
                      'VarSelectMechHierarchical_z_visualization']


def data_read(path):
    with open(path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    return load_dict


class Vocab:
    def __init__(self, vocab_file, vocab_size=50000, lang='ch', remove_stop=True, PATH=None, stop_words=None, dataset_name=None):
        self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3, '[MASK]': 4, '_TITLE_': 5}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]', '[MASK]', '_TITLE_']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1, '[MASK]': 1, '_TITLE_': 1}
        self.load_vocab(vocab_file, vocab_size)
        self.voc_size = len(self._word2id)
        self.UNK_token = 3
        self.PAD_token = 0
        self.PATH = PATH
        self.dataset_name = dataset_name

        if remove_stop:
            if lang == 'ch':
                print('loading chinese stopwords ...')
                self.stop_words = {word.strip() for word in open(os.path.join(self.PATH, stop_words)).readlines()}
            else:
                print('loading english stopwords ...')
                self.stop_words = {word.strip() for word in open(os.path.join(self.PATH, stop_words)).readlines()}
        else:
            self.stop_words = {}

    def load_vocab(self, vocab_file, vocab_size):
        for line in open(vocab_file):
            term_ = line.strip().split('\t')
            if len(term_) != 2:
                continue
            word, count = term_
            assert word not in self._word2id
            self._word2id[word] = len(self._word2id)
            self._id2word.append(word)
            self._wordcount[word] = int(count)
            if len(self._word2id) >= vocab_size:
                break
        assert len(self._word2id) == len(self._id2word)

    def word2id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        return self._word2id['[OOV]']

    def sent2id(self, sent, add_start=False, add_end=False, remove_stop=False):
        """
            add_start: is add the start flag
            add_end:  is add the end flag
            remove_stop: is remove the stop words
        """
        if not remove_stop:
            result = [self.word2id(word) for word in sent]
        else:
            result = [self.word2id(word) for word in sent if word not in self.stop_words]
        if add_start:
            result = [self._word2id['[START]']] + result

        if add_end:
            result = result + [self._word2id['[END]']]
        return result

    def id2word(self, word_id):
        return self._id2word[word_id]

    def id2sent(self, sent_id):
        result = []
        for id in sent_id:
            if id == self._word2id['[END]']:
                break
            elif id == self._word2id['[PADDING]']:
                continue
            result.append(self._id2word[id])
        return result


class Tokenizer():
    def __init__(self, vocab):
        super(Tokenizer, self).__init__()

        self.word2id = vocab._word2id
        self.id2word_list = vocab._id2word

        self.pad_token_id = vocab._word2id['[PADDING]']
        self.bos_token_id = vocab._word2id['[START]']
        self.eos_token_id = vocab._word2id['[END]']

        self.id2word = {}
        for index,word in enumerate(self.id2word_list):
            self.id2word.update({index: word})

    def convert_ids_to_tokens(self, ids, trim_bos=False, trim_pad=False, trim_from_eos=False, trim_after_eos=False):
        """
            id token --> word token
        """
        tokens = []
        for i in ids:
            if trim_bos and i == self.bos_token_id:
                continue
            if trim_pad and i == self.pad_token_id:
                continue
            if trim_from_eos and i == self.eos_token_id:
                break
            tokens.append(self.id2word[i])
            if trim_after_eos and i == self.eos_token_id:
                break
        return tokens

    def convert_tokens_to_string(self, tokens):
        """ 中/英文 空格方便切词  """
        sent = " ".join(tokens)
        return sent

    @staticmethod
    def toekn_processing(samples, tokenizer):
        hyps = []
        for sample in samples:
            hyp = sample.squeeze().tolist()
            hyp = tokenizer.convert_ids_to_tokens(ids=hyp,
                                                 trim_bos=True,
                                                 trim_from_eos=True,
                                                 trim_pad=True,
                                                 )
            hyp = tokenizer.convert_tokens_to_string(hyp)
            hyps.append(hyp)

        return hyps


class Example:
    """
    Each example is one data pair
        src: title (has oov)
        tgt: comment (oov has extend ids if use_oov else has oov)
        memory: tag (oov has extend ids)
    """

    def __init__(self, original_content, title, target, vocab, is_train, news_id, model, atricle_label=None, content_label=None, dataset_name=None):
        self.ori_title = title[:MAX_TITLE_LENGTH]
        self.ori_original_content = original_content[:MAX_ARTICLE_LENGTH]
        self.ori_news_id = news_id
        self.content_label = content_label
        self.atricle_label = atricle_label

        if content_label is not None:
            min_title = min(len(title), MAX_TITLE_LENGTH)
            temp = [l for l in content_label if l < min_title]
            min_body = min(len(original_content), MAX_ARTICLE_LENGTH)
            temp2 = [l for l in content_label if 0 <= l - len(title) < min_body]
            if MAX_TITLE_LENGTH < len(title):
                temp2 = [l - (len(title) - MAX_TITLE_LENGTH) for l in temp2]
            temp = temp + temp2
            if len(temp) == 0:
                temp = [-1]
            self.content_label = temp

        if is_train:
            self.ori_target = target[:MAX_COMMENT_LENGTH]
        else:
            self.ori_targets = [tar[:MAX_COMMENT_LENGTH] for tar in target]

        self.title = vocab.sent2id(self.ori_title)
        self.original_content = vocab.sent2id(self.ori_original_content)

        if is_train:
            self.target = vocab.sent2id(self.ori_target, add_start=True, add_end=True)

        if model in HierarchicalModels:
            """ Hierarchical model special processing code """
            self.sentence_content = split_chinese_sentence(original_content, dataset_name)
            self.sentence_content = [vocab.sent2id(sentence) for sentence in self.sentence_content]
            self.sentence_content_max_len = min(max([len(c) for c in self.sentence_content]), MAX_LENGTH)
            self.sentence_content, self.sentence_content_mask = Batch.padding(self.sentence_content, self.sentence_content_max_len, limit_length=True)

        elif model == 'topic_vae' or model == 'topic_and_saliency':
            if is_train:
                content_words = vocab.sent2id(self.ori_target, add_start=True, add_end=True, remove_stop=True)
                self.tgt_bow = np.bincount(content_words, minlength=vocab.voc_size)
                # normal
                self.tgt_bow[vocab.UNK_token] = 0
                self.tgt_bow = self.tgt_bow / np.sum(self.tgt_bow)

                self.ori_targets = [[tar[:MAX_COMMENT_LENGTH] for tar in target]]

        elif model == 'bow2seq':
            self.bow = self.bow_vec(self.original_content, MAX_ARTICLE_LENGTH)

    def bow_vec(self, content, max_len):
        bow = {}
        for word_id in content:
            if word_id not in bow:
                bow[word_id] = 0
            bow[word_id] += 1
        bow = list(bow.items())
        bow.sort(key=lambda k: k[1], reverse=True)
        bow.insert(0, (UNK, 1))
        return [word_id[0] for word_id in bow[:max_len]]


class Batch:
    """
    Each batch is a mini-batch of data

    """

    def __init__(self, example_list, is_train, model, update_size):
        self.model = model
        self.is_train = is_train
        self.examples = example_list
        self.update_size = update_size

        if model == 'topic_vae':
            if is_train:
                self.tgt_bow = torch.FloatTensor(np.array([e.tgt_bow for e in example_list]))
            else:
                title_list = [e.title for e in example_list]
                self.title_len = self.get_length(title_list, MAX_TITLE_LENGTH)
                self.title, self.title_mask = self.padding_list_to_tensor(title_list, self.title_len.max().item())

        elif model in HierarchicalModels:
            self.sentence_content = [np.array(e.sentence_content, dtype=np.long) for e in example_list]
            self.sentence_content_mask = [np.array(e.sentence_content_mask, dtype=np.int32) for e in example_list]
            self.sentence_content_len = [len(e.sentence_content) for e in example_list]
            max_sent_num = max(self.sentence_content_len)
            self.sentence_mask, _ = self.padding([[1 for _ in range(d)] for d in self.sentence_content_len], max_sent_num, limit_length=False)

            # from numpy to tensor
            self.sentence_content = [torch.from_numpy(src) for src in self.sentence_content]
            self.sentence_content_mask = [torch.from_numpy(mask) for mask in self.sentence_content_mask]
            self.sentence_content_len = torch.from_numpy(np.array(self.sentence_content_len, dtype=np.long))
            self.sentence_mask = torch.from_numpy(np.array(self.sentence_mask, dtype=np.int32))

            self.atricle_label = [e.atricle_label for e in example_list]

        elif model == 'graph2seq':
            self.src_len = [len(e.content) for e in example_list]
            batch_src = [e.content for e in example_list]
            self.src = [np.array(src, dtype=np.long) for src in batch_src]
            self.src_mask = [np.array(e.content_mask, dtype=np.int32) for e in example_list]
            concept_max_len = max([len(e.concept) for e in example_list])
            self.concept_vocab, self.concept_mask = self.padding([e.concept for e in example_list], concept_max_len)
            self.concept = [np.array(e.concept, dtype=np.long) for e in example_list]
            self.title_index = [e.title_index for e in example_list]
            self.adj = [e.adj for e in example_list]

            # from numpy to tensor
            self.src = [torch.from_numpy(src) for src in self.src]
            self.src_mask = [torch.from_numpy(mask) for mask in self.src_mask]
            self.src_len = torch.from_numpy(np.array(self.src_len, dtype=np.long))
            self.title_index = torch.from_numpy(np.array(self.title_index, dtype=np.long))
            self.concept = [torch.from_numpy(concept) for concept in self.concept]
            self.concept_vocab = torch.from_numpy(np.array(self.concept_vocab, dtype=np.long))
            self.concept_mask = torch.from_numpy(np.array(self.concept_mask, dtype=np.int32))

        elif model == 'bow2seq':
            bow_list = [e.bow for e in example_list]
            self.bow_len = self.get_length(bow_list, MAX_ARTICLE_LENGTH)
            self.bow, self.bow_mask = self.padding_list_to_tensor(bow_list, self.bow_len.max().item())

        else:
            # seq2seq, select_diverse2seq, select2seq and so on.
            content_list = [e.original_content for e in example_list]
            self.content_len = self.get_length(content_list, MAX_ARTICLE_LENGTH)
            self.content, self.content_mask = self.padding_list_to_tensor(content_list, self.content_len.max().item())

            title_list = [e.title for e in example_list]
            self.title_len = self.get_length(title_list, MAX_TITLE_LENGTH)
            self.title, self.title_mask = self.padding_list_to_tensor(title_list, self.title_len.max().item())

            title_content_list = [e.title + e.original_content for e in example_list]
            self.title_content_len = self.get_length(title_content_list, MAX_TITLE_LENGTH + MAX_ARTICLE_LENGTH)
            self.title_content, self.title_content_mask = self.padding_list_to_tensor(title_content_list, self.title_content_len.max().item())

            if model == 'topic_and_saliency':
                if is_train:
                    self.tgt_bow = torch.FloatTensor(np.array([e.tgt_bow for e in example_list]))

        if is_train:
            self.tgt_len = self.get_length([e.target for e in example_list])
            self.tgt, self.tgt_mask = self.padding_list_to_tensor([e.target for e in example_list], self.tgt_len.max().item())

            if example_list[0].content_label is not None:
                self.content_label = [e.content_label for e in example_list]

    @staticmethod
    def get_length(examples, max_len=1000):
        length = []
        for e in examples:
            if len(e) > max_len:
                length.append(max_len)
            else:
                length.append(len(e))
        assert len(length) == len(examples)
        length = torch.LongTensor(length)
        return length

    @staticmethod
    def padding_list_to_tensor(batch, max_len):
        padded_batch = []
        mask_batch = []
        for x in batch:
            y = x + [PAD] * (max_len - len(x))
            m = [1] * len(x) + [0] * (max_len - len(x))
            padded_batch.append(y)
            mask_batch.append(m)
        padded_batch = torch.LongTensor(padded_batch)
        mask_batch = torch.LongTensor(mask_batch).to(torch.uint8)
        return padded_batch, mask_batch

    @staticmethod
    def padding_2d_list_to_tensor(batch, max_len):
        padded_batch = []
        mask_batch = []
        for x in batch:
            y = x + [PAD] * (max_len - len(x))
            m = [1] * len(x) + [0] * (max_len - len(x))
            padded_batch.append(y)
            mask_batch.append(m)
        padded_batch = torch.LongTensor(padded_batch)
        mask_batch = torch.LongTensor(mask_batch).to(torch.uint8)
        return padded_batch, mask_batch

    @staticmethod
    def padding(batch, max_len, limit_length=True):
        """ the function come from Graph-2seq paper """
        if limit_length:
            max_len = min(max_len, MAX_LENGTH)
        result = []
        mask_batch = []
        for s in batch:
            l = copy.deepcopy(s)
            m = [1. for _ in range(len(l))]
            l = l[:max_len]
            m = m[:max_len]
            while len(l) < max_len:
                l.append(0)
                m.append(0.)
            result.append(l)
            mask_batch.append(m)
        return result, mask_batch


class DataLoader:
    def __init__(self, filename, config, vocab, model, is_train=True, debug=False, train_num=0, is_vaild=False):
        self.batch_size = 1 if is_vaild else config.batch_size

        self.vocab = vocab
        self.config = config
        # self.max_len = MAX_LENGTH

        self.filename = filename

        # 从磁盘中读取数据
        if self.config.dataset_name == 'NeteaseComment':
            self.NeteaseComment_dicts = data_read(self.filename)
            self.sample_num = len(self.NeteaseComment_dicts)
        else:
            self.stream = open(self.filename, encoding='utf8')
            self.sample_num = len(self.stream.readlines())

        self.epoch_id = 0
        self.is_train = is_train
        self.debug = debug
        self.train_num = train_num

        self.model = model

        if self.config.dataset_name == 'yahoo':
            self.create_comments_from_article = lambda x, y: self.create_comments_from_yahoo_article(x, y)

        elif self.config.dataset_name == '163':
            self.create_comments_from_article = lambda x, y: self.create_comments_from_163_article(x, y)

        elif self.config.dataset_name == 'tencent':
            self.create_comments_from_article = lambda x, y: self.create_comments_from_tencent_article(x, y)

        elif self.config.dataset_name == 'kuibao':
            self.create_comments_from_article = lambda x, y: self.create_comments_from_kuibao_article(x, y)

        elif self.config.dataset_name == 'cross_modal':
            self.create_comments_from_article = lambda x, y: self.create_comments_from_cross_modal_article(x, y)

        elif self.config.dataset_name == 'NeteaseComment':
            self.create_comments_from_article = lambda x, y: self.create_comments_from_NeteaseComment_article(x, y)

        else:
            raise FileNotFoundError(f'{self.config.dataset_name} Dataset not exist!')



    def __iter__(self):
        articles = []

        if self.config.dataset_name == 'NeteaseComment':
            articles = self.NeteaseComment_dicts

        else:
            lines = self.stream.readlines()

            # next epoch
            if not lines:
                self.epoch_id += 1
                self.stream.close()
                self.stream = open(self.filename, encoding='utf8')
                lines = self.stream.readlines()


            for line in lines:
                line = line.strip()
                if not line:
                    continue
                articles.append(json.loads(line))

                if len(articles) > 100 and self.debug:
                    break
                if self.train_num > 0:
                    if len(articles) >= self.train_num:
                        break

        data = []
        for idx, doc in enumerate(articles):
            data.extend(self.create_comments_from_article(doc, None))

        if self.is_train:
            random.shuffle(data)

        # 计算数据迭代多少次
        self.update_size = math.ceil(len(data)/self.batch_size)

        idx = 0
        while idx < len(data):
            example_list = self.covert_json_to_example(data[idx:idx + self.batch_size])
            yield Batch(example_list, self.is_train, self.model, self.update_size)
            idx += self.batch_size

    def create_comments_from_tencent_article(self, article, article_ext=None):
        comments = []
        if self.is_train:
            for i in range(len(article['comment'])):
                item = dict()
                item['title'] = article['title']
                item['body'] = article['body']
                item['comment'] = article['comment'][i][0]
                comments.append(item)

                if article_ext is not None:
                    item['content_label'] = article_ext['com_labels'][i]

                if 0 < self.config.max_comment_num <= len(comments):
                    break
        else:
            # contain one article and multi comments
            item = dict()
            # item['id'] = article['id']
            item['title'] = article['title']
            item['body'] = article['body']
            item['comment'] = [c[0] for c in article['comment'][:5]]
            comments.append(item)
        return comments

    def create_comments_from_yahoo_article(self, article, article_ext=None):
        comments = []
        if self.is_train:
            for i in range(len(article['cmts'])):
                item = dict()
                item['title'] = article['title'].lower()
                item['body'] = ' '.join(article['paras']).lower()
                item['comment'] = article['cmts'][i]['cmt'].lower()
                comments.append(item)

                if article_ext is not None:
                    item['content_label'] = article_ext['com_labels'][i]

                if 0 < self.config.max_comment_num <= len(comments):
                    break
        else:
            # contain one article and multi comments
            item = dict()
            item['id'] = article['_id']
            item['title'] = article['title']
            item['body'] = ' '.join(article['paras'])
            item['comment'] = [c['cmt'] for c in article['cmts'][:5]]
            comments.append(item)
        return comments

    def create_comments_from_163_article(self, article, article_ext=None):
        comments = []
        if self.is_train:
            for i in range(len(article['cmts'])):
                item = dict()
                item['title'] = article['title']
                item['body'] = article['body']
                item['comment'] = article['cmts'][i]['cmt']
                comments.append(item)

                if article_ext is not None:
                    item['content_label'] = article_ext['com_labels'][i]

                if 0 < self.config.max_comment_num <= len(comments):
                    break
        else:
            # contain one article and multi comments
            item = dict()
            item['id'] = article['_id']
            item['title'] = article['title']
            item['body'] = article['body']
            item['comment'] = [c['cmt'] for c in article['cmts'][:5]]
            comments.append(item)
        return comments

    def create_comments_from_kuibao_article(self, article, article_ext=None):
        """ 原始数据集已经做了预处理，即一条新闻对应一条评论 """
        comments = []

        item = dict()
        item['title'] = article['title']
        item['body'] = article['text']

        if self.is_train:
            item['comment'] = article['label']
        else:
            item['comment'] = [article['label']]

        comments.append(item)

        return comments

    def create_comments_from_cross_modal_article(self, article, article_ext=None):
        """ 原始数据集已经做了预处理，即一条新闻对应一条评论 """
        comments = []

        item = dict()
        item['title'] = article['title']
        item['body'] = ' '

        if self.is_train:
            item['comment'] = article['comment']
        else:
            item['comment'] = [article['comment']]

        comments.append(item)

        return comments

    def create_comments_from_NeteaseComment_article(self, article, article_ext=None):
        comments = []

        item = dict()
        item['title'] = article['title']
        item['body'] = ' '.join(article['content'])     # 句子之间用空格划分
        item['atricle_label'] = article['label']

        if self.is_train:
            item['comment'] = list(article['comment'].keys())[-1]
        else:
            item['comment'] = [list(article['comment'].keys())[-1]]

        comments.append(item)

        return comments

    def covert_json_to_example(self, json_list):
        results = []
        for g in json_list:
            if self.is_train:
                target = g['comment'].split()
            else:
                # multi comments for each article
                target = [s.split() for s in g['comment']]

            title = g["title"].split()
            original_content = g["body"].split()

            atricle_label = g['atricle_label'] if 'atricle_label' in g else None
            content_label = g['content_label'] if 'content_label' in g else None

            # news_id = None if self.is_train else g["id"]
            news_id = None

            e = Example(original_content, title, target,
                        self.vocab, self.is_train, news_id=news_id,
                        model=self.model, atricle_label=atricle_label, 
                        content_label=content_label, dataset_name=self.config.dataset_name)
            results.append(e)
        return results

