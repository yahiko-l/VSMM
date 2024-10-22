import codecs
import csv
import os
import time
import json

import yaml
import torch
import torch.nn as nn

from sacrebleu.metrics import BLEU

from utils.metrics.metric import calc_rouge_l
from utils.metrics.metric import calc_meteor
from utils.metrics.metric import cal_diversity_n
from utils.metrics.metric import calc_Self_BLEU


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.safe_load(open(path, 'r')))


def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s)

    return write_log


def logging_csv(file):
    def write_csv(s):
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(s)

    return write_csv


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)


def single_comment_generation_metric(examples, candidates, logging_csv,log_path, epoch, updates, vocab):
    """
        评估 single_comment_generation
    """
    assert len(examples) == len(candidates), (len(examples), len(candidates))

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path = log_path.strip('/')
    log_file = log_path + '/observe_result.tsv.%d.json' % epoch
    outputs = []

    total_rouge_l = 0.
    total_meteor = 0.

    total_Forward_BLEU = 0.
    total_Backward_BLEU = 0.

    count = 0

    # instantiation BLEU function with zh language
    bleu = BLEU(tokenize='zh')

    # modification '[]' --> '[，]' list in candidates, avoid error
    candidates = [x if x != [] else ['[]'] for x in candidates]

    # 无间隔的数据 [[gen1],[gen2]], 用于计算self-belu
    gen_comments_bleu = [["".join(cand).strip()] for _, cand in zip(examples, candidates)]

    dataset_name = vocab.dataset_name
    # 计算每条句子的 Harmonic_BLEU、Rouge-L、 METEOR 然后再求平均值
    for e, cand in zip(examples, candidates):
        if dataset_name == 'yahoo':
            out_dict = {'title':" ".join(e.ori_title).strip(),
                        'body':" ".join(e.ori_original_content).strip(),
                        'comment':[" ".join(comment).strip()  for comment in e.ori_targets],
                        'auto_comment':" ".join(cand).strip()}
        else:
            out_dict = {'title':"".join(e.ori_title).strip(),
                        'body':"".join(e.ori_original_content).strip(),
                        'comment':["".join(comment).strip()  for comment in e.ori_targets],
                        'auto_comment':"".join(cand).strip()}
        outputs.append(out_dict)

        """ metric preprocesssing  data"""
        # bleu 计算时，采用自己的 token 划定，所以comment是一条无间隔句子
        gen_comment_bleu = ["".join(cand).strip()]
        ref_comments_bleu = [["".join(comment).strip()] for comment in e.ori_targets]

        # 计算rouge使用， 每个词需要 空格 划分
        gen_comment_rouge = [" ".join(cand).strip()]
        ref_comments_rouge = [[" ".join(comment).strip()] for comment in e.ori_targets]

        # 计算的 METEOR 每个词单独单独划分
        gen_comment_meteor = cand
        ref_comments_meteor = [comment for comment in e.ori_targets]

        # 计算 Harmonic_BLEU、Rouge
        """
            针对 from sacrebleu.metrics import BLEU 这个包，我们采用 bleu = BLEU(tokenize='zh')
            所以

            输入的格式如下：
            corpus_score(hypotheses, references)
            hypotheses = [single_sentence]
            references = [[reference1], [reference2], ..., [reference_n]]

            computing Forward BLEU and Backward BLEU
            Forward BLEU: It measures the precision (fluency) of the generator
            Backward BLEU: It measures the recall (diversity) of the generator

            ref_comments 可以采用文章中的5条评论，但目前的数据仅测试集只有一个reference

            auto_comment = ['中国土狗。好养，抵抗力好，顾家，通人性']
            ref_comments = [['我准备要养一只中华了，狗眼不看人低，很多是人眼看狗低罢了'], ['我天天遛土狗呢！怎么滴怎么滴怎么滴']]
        """
        Forward_BLEU = bleu.corpus_score(gen_comment_bleu, ref_comments_bleu)        # 41.11
        total_Forward_BLEU += Forward_BLEU.score

        # 去掉相同的评论
        backward_auto_comments = []
        for i in gen_comments_bleu:
            if gen_comment_bleu != i:
                backward_auto_comments.append(i)
        Backward_BLEU = bleu.corpus_score(gen_comment_bleu, backward_auto_comments) # 51.37
        total_Backward_BLEU += Backward_BLEU.score

        """ computing Rouge-L
            auto_comment = ['中国 土狗 。 好 养 ， 抵抗力 好 ，顾 家，通 人性']
            ref_comments = [['我 准备 要 养 一只 中华 了  ，狗 眼 不 看 人 低 ， 很多 是 人 眼 看 狗 低 罢 了']]
        """
        rouge_l_score = calc_rouge_l(gen_comment_rouge, ref_comments_rouge)                # 8.11
        total_rouge_l += rouge_l_score

        """ computing METEOR
            hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
            references = [['It', 'is', 'a', 'cat', 'inside', 'the', 'room']]
        """
        meteor_score = calc_meteor(gen_comment_meteor, ref_comments_meteor)                  # 3.09
        total_meteor += meteor_score

        count += 1

    json.dump(outputs, open(log_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    # harmonic average total_Forward_BLEU and total_Backward_BLEU
    Harmonic_BLEU = (2 * total_Forward_BLEU * total_Backward_BLEU) / (total_Forward_BLEU + total_Backward_BLEU)

    """ computing average socre of bleu  Rouge-L METEOR"""
    logging_csv(['avg-Harmonic_BLEU', round(Harmonic_BLEU/count, 2)])
    logging_csv(['avg-Rouge-L', round(total_rouge_l/count, 2)])
    logging_csv(['avg-METEOR', round(total_meteor/count, 2)])

    output_scores = 'avg-Harmonic_BLEU:%.2f | avg-Rouge-L:%.2f | avg-METEOR:%.2f ' % (
        Harmonic_BLEU/count, total_rouge_l/count, total_meteor/count,
        )
    print(output_scores)


    """ computing diversity of generate text """
    # computing Self-BLEU
    avg_self_bleu = calc_Self_BLEU(gen_comments_bleu)
    logging_csv(['avg_self_bleu', avg_self_bleu])
    print('avg_self_bleu:', avg_self_bleu)

    # computing Distince-N
    diversity_n = cal_diversity_n(candidates, logging_csv, epoch, updates)
    diversity_n = diversity_n.split(',')[:4]
    diversities = []
    for diversity in diversity_n:
        diversity = float(diversity.split('=')[-1])
        diversities.append(round(diversity, 2))

    logging_csv(['diversity-1', diversities[0]])
    logging_csv(['diversity-2', diversities[1]])
    logging_csv(['diversity-3', diversities[2]])
    logging_csv(['diversity-4', diversities[3]])

    print('diversity_n: ', diversity_n)

    # wirte condidates to txt files for observation.s
    log_file = log_path + '/result_for_test.tsv.%d.txt' % epoch
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            f.write(str(e.ori_news_id) + '\t')
            f.write(" ".join(cand).strip())
            f.write("\n")


def multi_comment_generation_metric(examples, candidates, logging_csv,log_path, epoch, updates, beam_size):
    """
        评估 multi_comment_generation
        计算多样性时为全域文本，这样会导致非常耗时
    """
    assert len(examples) * beam_size == len(candidates), (len(examples), len(candidates))

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path = log_path.strip('/')
    log_file = log_path + '/multi_comment_generation-observe_result.tsv.%d.json' % epoch
    outputs = []
    auto_comments = []

    # modification '[]' --> '[，]' list in candidates, avoid error
    candidates = [x if x != [] else ['[]'] for x in candidates]

    # 用于展示beam search结果用
    beam_candidates = []
    start = 0
    for i in range(0,len(candidates), beam_size):
        if i != len(candidates):
            sub_candidates = candidates[start: start + beam_size]
            sub_candidates = [''.join(sub_candidate) for sub_candidate in sub_candidates]
            beam_candidates.append(sub_candidates)
            start = start + beam_size

    for e, cand in zip(examples, beam_candidates):
        out_dict = {'title':"".join(e.ori_title).strip(),
                    'body':"".join(e.ori_original_content).strip(),
                    'comment':["".join(comment).strip()  for comment in e.ori_targets],
                    'auto_comment':cand}

        outputs.append(out_dict)
        """ metric preprocesssing  data"""
        # auto_comment = ["".join(cand).strip()]
        # auto_comments.append(auto_comment)

    json.dump(outputs, open(log_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    gen_comment_bleu = [[''.join(candidate)] for candidate in candidates]

    """ computing diversity of generate text """
    # computing Self-BLEU
    avg_self_bleu = calc_Self_BLEU(gen_comment_bleu)
    logging_csv(['avg_self_bleu', avg_self_bleu])
    print('avg_self_bleu:', avg_self_bleu)


    # computing Distince-N
    diversity_n = cal_diversity_n(candidates, logging_csv, epoch, updates)
    diversity_n = diversity_n.split(',')[:4]
    diversities = []
    for diversity in diversity_n:
        diversity = float(diversity.split('=')[-1])
        diversities.append(round(diversity, 2))

    logging_csv(['diversity-1', diversities[0]])
    logging_csv(['diversity-2', diversities[1]])
    logging_csv(['diversity-3', diversities[2]])
    logging_csv(['diversity-4', diversities[3]])

    print('diversity_n: ', diversity_n)

    # wirte condidates to txt files for observation.s
    log_file = log_path + '/result_for_test.tsv.%d.txt' % epoch
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            f.write(str(e.ori_news_id) + '\t')
            f.write(" ".join(cand).strip())
            f.write("\n")


def multi_comment_generation_metric_with_beam(examples, candidates, logging_csv,log_path, epoch, updates, config):
    """
        评估 multi_comment_generation
        计算多样性时，参考文本为 beam_size内的文本
    """
    VarSelectMechModels = ['VarSelectMech', 'VarSelectMechHierarchical']
    if config.model in VarSelectMechModels and config.ranking_type == 'full':
        beam_size = config.num_mappings
    else:
        beam_size = config.beam_size

    # assert len(examples) * beam_size == len(candidates)

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path = log_path.strip('/')
    log_file = log_path + '/multi_comment_generation-observe_result.tsv.%d.json' % epoch
    outputs = []
    total_self_bleu = 0.0
    count = 0

    # modification '[]' --> '[，]' list in candidates, avoid error
    candidates = [x if x != [] else ['[]'] for x in candidates]

    # 用于展示beam search结果用
    beam_candidates = []
    start = 0
    for i in range(0,len(candidates), beam_size):
        if i != len(candidates):
            sub_candidates = candidates[start: start + beam_size]
            sub_candidates = [[''.join(sub_candidate)] for sub_candidate in sub_candidates]
            beam_candidates.append(sub_candidates)
            start = start + beam_size

    for e, cand in zip(examples, beam_candidates):
        out_dict = {'title':"".join(e.ori_title).strip(),
                    'body':"".join(e.ori_original_content).strip(),
                    'comment':["".join(comment).strip()  for comment in e.ori_targets],
                    'auto_comment':cand}


        # computing Self-BLEU， 采用beam_size 内计算
        self_bleu = calc_Self_BLEU(cand)
        total_self_bleu += self_bleu


        outputs.append(out_dict)
        """ metric preprocesssing  data"""
        # auto_comment = ["".join(cand).strip()]
        # auto_comments.append(auto_comment)

        count += 1

    json.dump(outputs, open(log_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


    """ computing diversity of generate text """

    logging_csv(['avg_self_bleu', round(total_self_bleu/count, 2)])
    print('avg_self_bleu:', round(total_self_bleu/count, 2))

    # computing Distince-N， 采用全域文本计算
    diversity_n = cal_diversity_n(candidates, logging_csv, epoch, updates)
    diversity_n = diversity_n.split(',')[:4]
    diversities = []
    for diversity in diversity_n:
        diversity = float(diversity.split('=')[-1])
        diversities.append(round(diversity, 2))

    logging_csv(['diversity-1', diversities[0]])
    logging_csv(['diversity-2', diversities[1]])
    logging_csv(['diversity-3', diversities[2]])
    logging_csv(['diversity-4', diversities[3]])

    print('diversity_n: ', diversity_n)

    # wirte condidates to txt files for observation.s
    log_file = log_path + '/result_for_test.tsv.%d.txt' % epoch
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            f.write(str(e.ori_news_id) + '\t')
            f.write(" ".join(cand).strip())
            f.write("\n")


def multi_comment_generation_quality(examples, candidates, logging_csv,log_path, epoch, updates, vocab):
    """
        评估 single_comment_generation
    """
    assert len(examples) == len(candidates), (len(examples), len(candidates))

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path = log_path.strip('/')
    log_file = log_path + '/observe_result.tsv.%d.json' % epoch
    outputs = []

    total_rouge_l = 0.
    total_meteor = 0.

    count = 0

    # modification '[]' --> '[，]' list in candidates, avoid error
    candidates = [x if x != [] else ['[]'] for x in candidates]

    dataset_name = vocab.dataset_name
    # 计算每条句子的 Harmonic_BLEU、Rouge-L、 METEOR 然后再求平均值
    for e, cand in zip(examples, candidates):
        if dataset_name == 'yahoo':
            out_dict = {'title':" ".join(e.ori_title).strip(),
                        'body':" ".join(e.ori_original_content).strip(),
                        'comment':[" ".join(comment).strip()  for comment in e.ori_targets],
                        'auto_comment':" ".join(cand).strip()}
        else:
            out_dict = {'title':"".join(e.ori_title).strip(),
                        'body':"".join(e.ori_original_content).strip(),
                        'comment':["".join(comment).strip()  for comment in e.ori_targets],
                        'auto_comment':"".join(cand).strip()}
        outputs.append(out_dict)

        """ metric preprocesssing  data"""
        # 计算rouge使用， 每个词需要 空格 划分
        gen_comment_rouge = [" ".join(cand).strip()]
        ref_comments_rouge = [[" ".join(comment).strip()] for comment in e.ori_targets]

        # 计算的 METEOR 每个词单独单独划分
        gen_comment_meteor = cand
        ref_comments_meteor = [comment for comment in e.ori_targets]

        """ computing Rouge-L
            auto_comment = ['中国 土狗 。 好 养 ， 抵抗力 好 ，顾 家，通 人性']
            ref_comments = [['我 准备 要 养 一只 中华 了  ，狗 眼 不 看 人 低 ， 很多 是 人 眼 看 狗 低 罢 了']]
        """
        rouge_l_score = calc_rouge_l(gen_comment_rouge, ref_comments_rouge)                # 8.11
        total_rouge_l += rouge_l_score

        """ computing METEOR
            hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
            references = [['It', 'is', 'a', 'cat', 'inside', 'the', 'room']]
        """
        meteor_score = calc_meteor(gen_comment_meteor, ref_comments_meteor)                  # 3.09
        total_meteor += meteor_score

        count += 1

    json.dump(outputs, open(log_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


    """ computing average socre of bleu  Rouge-L METEOR"""
    logging_csv(['avg-Rouge-L', round(total_rouge_l/count, 2)])
    logging_csv(['avg-METEOR', round(total_meteor/count, 2)])

    output_scores = 'avg-Rouge-L:%.2f | avg-METEOR:%.2f ' % (
        total_rouge_l/count, total_meteor/count,
        )
    print(output_scores)


    # wirte condidates to txt files for observation.s
    log_file = log_path + '/result_for_test.tsv.%d.txt' % epoch
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            f.write(str(e.ori_news_id) + '\t')
            f.write(" ".join(cand).strip())
            f.write("\n")


def write_result_to_file(examples, candidates, log_path, epoch):
    assert len(examples) == len(candidates), (len(examples), len(candidates))

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path = log_path.strip('/')
    log_file = log_path + '/observe_result.tsv.%d' % epoch
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            f.write("".join(cand).strip() + '\t')
            f.write("".join(e.ori_title).strip() + '\t')
            # f.write(";".join(["".join(sent).strip() for sent in e.ori_content]) + '\t')
            f.write("".join(e.ori_original_content).strip() + '\t')
            # f.write("$$".join(["".join(comment).strip() for comment in e.ori_targets]) + '\t')
            f.write("\n")

    log_file = log_path + '/result_for_test.tsv.%d' % epoch
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            f.write(str(e.ori_news_id) + '\t')
            f.write(" ".join(cand).strip())
            f.write("\n")


def write_multi_result_to_file(examples, candidates, log_path, epoch, args, file_sufix=''):
    assert len(examples) == len(candidates), (len(examples), len(candidates))

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path = log_path.strip('/')
    log_file = log_path + '/observe_result.tsv.beam.%d.%s' % (epoch, file_sufix)
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            cand_str = list(map(lambda com: ''.join(com).strip(), cand))
            f.write(" <sep> ".join(cand_str).strip() + '\t')
            f.write("".join(e.ori_title).strip() + '\t')
            # f.write(";".join(["".join(sent).strip() for sent in e.ori_content]) + '\t')
            f.write("".join(e.ori_original_content).strip() + '\t')
            # f.write("$$".join(["".join(comment).strip() for comment in e.ori_targets]) + '\t')
            f.write("\n")

    beam_size = len(candidates[0])
    log_files = [log_path + '/result_for_test.tsv.beam.%d.%d.%s' % (epoch, i, file_sufix) for i in range(beam_size)]
    fs = [codecs.open(log, 'w', 'utf-8') for log in log_files]
    for e, cand in zip(examples, candidates):
        for ii, f in enumerate(fs):
            f.write(str(e.ori_news_id) + '\t')

            # modify by yahiko
            if args.beam_search:
                f.write(" ".join(cand[ii]).strip())
            else:
                f.write(" ".join(cand).strip())

            f.write("\n")


def write_multi_result_to_dict_file(examples, candidates, log_path, epoch, args, file_sufix=''):
    assert len(examples) == len(candidates), (len(examples), len(candidates))

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path = log_path.strip('/')
    log_file = log_path + '/observe_result.tsv.beam.%d.%s.json' % (epoch, file_sufix)
    outputs = []

    for e, cand in zip(examples, candidates):
        out_dict = {
                    'title':"".join(e.ori_title).strip(),
                    'body':"".join(e.ori_original_content).strip(),
                    'comment':["".join(comment).strip() for comment in e.ori_targets],
                    'auto_comment':["".join(auto_comment).strip() for auto_comment in cand]
                    }
        outputs.append(out_dict)
    json.dump(outputs, open(log_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    beam_size = len(candidates[0])
    log_files = [log_path + '/result_for_test.tsv.beam.%d.%d.%s' % (epoch, i, file_sufix) for i in range(beam_size)]
    fs = [codecs.open(log, 'w', 'utf-8') for log in log_files]

    for e, cand in zip(examples, candidates):
        for ii, f in enumerate(fs):
            f.write(str(e.ori_news_id) + '\t')

            # modify by yahiko
            if args.beam_search:
                f.write(" ".join(cand[ii]).strip())
            else:
                f.write(" ".join(cand).strip())

            f.write("\n")


def write_topic_result_to_file(examples, candidates, log_path, epoch, topic, data_type='topic', file_sufix=''):
    assert len(examples) == len(candidates), (len(examples), len(candidates))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = log_path.strip('/')
    log_file = log_path + '/result_for_test.tsv.%s.%d.%d.%s' % (data_type, epoch, topic, file_sufix)
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            f.write(str(e.ori_news_id) + '\t')
            f.write(" ".join(cand).strip())
            f.write("\n")


def write_observe_to_file(examples, candidates, log_path, epoch, dataset, file_sufix=''):
    assert len(examples) == len(candidates), (len(examples), len(candidates))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    sep = ' ' if dataset == 'yahoo' else ''

    log_path = log_path.strip('/')
    log_file = log_path + '/observe_result.tsv.topic.%d.%s' % (epoch, file_sufix)
    with codecs.open(log_file, 'w', 'utf-8') as f:
        for e, cand in zip(examples, candidates):
            cand_str = list(map(lambda com: sep.join(com).strip(), cand))
            f.write(" <sep> ".join(cand_str).strip() + '\t')
            f.write(sep.join(e.ori_title).strip() + '\t')
            # f.write(";".join(["".join(sent).strip() for sent in e.ori_content]) + '\t')
            f.write(sep.join(e.ori_original_content).strip() + '\t')
            # f.write("$$".join(["".join(comment).strip() for comment in e.ori_targets]) + '\t')
            f.write("\n")


def save_model(path, model, optim, epoch, updates, score, config, args):
    model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'epcoh': epoch,
        'updates': updates,
        'best_eval_score': score
    }
    torch.save(checkpoints, path)


def total_parameters(model, logging):
    logging(repr(model) + "\n\n")
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    logging('total number of parameters: %d\n\n' % param_count)


def split_chinese_sentence(text, dataset_name):
    """
    Segment a input Chinese text into a list of sentences.
    :param text: a segmented input string.
    :return: a list of segmented sentences.
    """
    if type(text) == list:  # already segmented
        words = text
    else:
        words = str(text).split()
    start = 0
    i = 0
    sents = []

    # 由于我们提出的数据 NeteaseComment 是提前处理好了句子划分，而先前句子划分符号会导致句子划分出问题。
    if dataset_name == 'NeteaseComment':
        punt_list = '。!！?？'
    else:
        punt_list = '。!！?？;；~～'

    for word in words:
        word = word
        token = list(words[start:i + 2]).pop()
        if word in punt_list and token not in punt_list:
            sents.append(words[start:i + 1])
            start = i + 1
            i += 1
        else:
            i += 1
            token = list(words[start:i + 2]).pop()
    if start < len(words):
        sents.append(words[start:])
    sents = [" ".join(x) for x in sents]
    return sents


def init_module_weights(m, init_w=0.08):
    if isinstance(m, (nn.Parameter)):
        m.data.uniform_(-1.0*init_w, init_w)
    elif isinstance(m, (nn.Linear, nn.Bilinear)):
        m.weight.data.uniform_(-1.0*init_w, init_w)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Embedding):
        m.weight.data.uniform_(-1.0, 1.0)
    elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
    elif isinstance(m, nn.MultiheadAttention):
        for param in m.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    elif isinstance(m, nn.ModuleDict):
        for submodule in m.values():
            init_module_weights(submodule)
    elif isinstance(m, (nn.ModuleList, nn.Sequential)):
        for submodule in m:
            init_module_weights(submodule)
    elif isinstance(m, (nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Dropout)):
        pass
    elif isinstance(m, (nn.BatchNorm1d)):
        pass
    elif isinstance(m, (nn.Identity)):
        pass
    else:
        raise Exception("undefined initialization for module {}".format(m))


def embedded_dropout(embed, words, dropout, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse)
    return X


def init_rnn_hidden_states(batch_size, hidden_dim, n_layers, bidirectional,
                           rnn_type, init_type="zero"):
    n_directions = 2 if bidirectional else 1
    hidden_state_size = (
        n_directions*n_layers,
        batch_size,
        hidden_dim
    )

    def init_vec(size, init_type):
        if init_type == "zero":
            return torch.FloatTensor(*size).zero_().to(DEVICE)
        elif init_type == "uniform":
            return torch.FloatTensor(*size).uniform_(-1.0, 1.0).to(DEVICE)

    if rnn_type == "lstm":
        hiddens = (
            init_vec(hidden_state_size, init_type),
            init_vec(hidden_state_size, init_type),
        )
    else:
        hiddens = init_vec(hidden_state_size, init_type)
    return hiddens


def gaussian_kld(mu1, var1,
                 mu2=torch.FloatTensor([0.0]).to(DEVICE),
                 var2=torch.FloatTensor([1.0]).to(DEVICE),
                 reduction="sum"):
    """ 高斯KLD计算方式 """

    one = torch.FloatTensor([1.0]).to(DEVICE)
    losses = 0.5 * (torch.log(var2) - torch.log(var1) + (var1 + (mu1 - mu2).pow(2)) / var2 - one)
    if reduction == "sum":
        return losses.sum(1)
    elif reduction == "None":
        return losses
    else:
        raise Exception(f"Unexpected reduction type {reduction}.")


def normal_kl_div(mu1,
                  var1,
                  mu2=torch.FloatTensor([0.0]).to(DEVICE),
                  var2=torch.FloatTensor([1.0]).to(DEVICE)):
    one = torch.FloatTensor([1.0]).to(DEVICE)
    return torch.sum(0.5 * (torch.log(var2) - torch.log(var1) + (var1 + (mu1 - mu2).pow(2)) / var2 - one), -1)


def gaussian_kld2(recog_mu, recog_logvar, prior_mu, prior_logvar):
    """ no log function for var """
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)

    return kld.sum(-1)


def init_word_embedding(load_pretrained_word_embedding=False,
                        pretrained_word_embedding_path=None,
                        specific_pretrained_word_embedding_path=None,
                        id2word=None,
                        word_embedding_dim=None,
                        vocab_size=None,
                        pad_token_id=None):
    if load_pretrained_word_embedding:
        # 载入预训练词向量
        embeddings = []
        # specific_dict = {}

        pretrained_embeddings = json.load(open(pretrained_word_embedding_path))
        in_vocab_cnt = 0
        for word_id in range(len(id2word)):
            word = id2word[word_id]
            if word in pretrained_embeddings:
                embeddings.append(pretrained_embeddings[word])
                in_vocab_cnt += 1
            else:
                embeddings.append([0.0]*word_embedding_dim)

            # specific_dict.update({word: embeddings[word_id]})
        weights = nn.Parameter(torch.FloatTensor(embeddings).to(DEVICE))
        print("{}/{} pretrained word embedding in vocab".format(in_vocab_cnt, vocab_size))

        # # add the specific dict
        # with open(specific_pretrained_word_embedding_path, "w", encoding="utf-8") as f:
        #     json.dump(specific_pretrained_word_embedding_path, f)
    else:
        # 随机初始化
        weights = nn.Parameter(
            torch.FloatTensor(
                vocab_size,
                word_embedding_dim
            ).to(DEVICE)
        )
        torch.nn.init.uniform_(weights, -1.0, 1.0)

    weights[pad_token_id].data.fill_(0)
    return weights



