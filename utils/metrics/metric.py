from collections import OrderedDict
from curses import beep

from numpy import average


def calc_diversity(texts):
    """ single sentence calc distinct， 单位为 %"""
    unigram, bigram, trigram, qugram = set(), set(), set(), set()
    num_tok = 0
    for vec in texts:
        v_len = len(vec)
        num_tok += v_len
        unigram.update(vec)
        bigram.update([tuple(vec[i:i + 2]) for i in range(v_len - 1)])
        trigram.update([tuple(vec[i:i + 3]) for i in range(v_len - 2)])
        qugram.update([tuple(vec[i:i + 4]) for i in range(v_len - 3)])

    metrics = OrderedDict()
    metrics['dist_1'] = round((len(unigram) * 1.0 / num_tok) * 100, 4)
    metrics['dist_2'] = round((len(bigram) * 1.0 / num_tok) * 100, 4)
    metrics['dist_3'] = round((len(trigram) * 1.0 / num_tok) * 100, 4)
    metrics['dist_4'] = round((len(qugram) * 1.0 / num_tok) * 100, 4)

    # 暂时只显示 distin-1, -2, -3, -4
    metrics['num_d1'] = len(unigram)
    metrics['num_d2'] = len(bigram)
    metrics['num_d3'] = len(trigram)
    metrics['num_d4'] = len(qugram)

    metrics['total_num'] = num_tok
    metrics['avg_sen_len'] = round(num_tok * 1.0 / len(texts), 4)

    return metrics


def cal_diversity_n(candidate, logging_csv, epoch, updates):
    # distinct, best 1 and best n
    """ 单个句子测试 """
    # single_candidate = [c[0] for c in candidate]
    # metrics_best_1 = calc_diversity(single_candidate)
    # text_result = ','.join('{:s}={:.6f}'.format(key, metrics_best_1[key]) for key in metrics_best_1.keys())
    # logging_csv([epoch, updates, text_result])
    # print(text_result, flush=True)

    """ 多个句子测试 """
    flatten_candidate = [si for can in candidate for si in can]
    metrics_best_n = calc_diversity(flatten_candidate)
    text_result = ','.join('{:s}={:.4f}'.format(key, metrics_best_n[key]) for key in metrics_best_n.keys())

    return text_result


def calc_rouge_l(auto_comment, ref_comments):
    from .rouge_zh.rouge_zh import Rouge
    """
    这里建议应该取 auto_comment 多个ref_comments计算后，应该取最大值才合理，
    但这样的方式存在一些问题。出现简单的评论与生成评论很相似，导致评论分很高，
    特别是seq2seq模型，可以采用惩罚因子，降低score 后续的研究工作。

    return: max_rouge_l value [0-100]
    单位为 %

    auto_comment = ['中国 土狗 。 好 养 ， 抵抗力 好 ，顾 家，通 人性']
    ref_comments = [['我 准备 要 养 一只 中华 了  ，狗 眼 不 看 人 低 ， 很多 是 人 眼 看 狗 低 罢 了']]

    """
    max_f = 0.
    for ref_comment in ref_comments:
        auto_comment = [" ".join(list(auto_comment[0]))]
        ref_comment = [[" ".join(list(ref_comment[0]))]]
        
        rouge_l  = Rouge(metrics=['rouge-l'],
                            length_limit_type='words',
                            alpha=0.5, # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

        rouge_score = rouge_l.get_scores(auto_comment, ref_comment)
        f = rouge_score['rouge-l']['f']

        if f >= max_f:
            max_rouge_l = round(f*100, 2)

    return max_rouge_l


def calc_meteor(auto_comment, ref_comments):
    """
        hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
        references = [['It', 'is', 'a', 'cat', 'inside', 'the', 'room']]

        meteor 不采用char划分，采用词划分后评估

        return:meteor_score value [0-100]
        单位为 %
    """
    from nltk.translate.meteor_score import meteor_score as Meteor
    
    meteor_score = round(Meteor(ref_comments, auto_comment)*100, 2)

    return meteor_score


def calc_Self_BLEU(auto_comments):
    """
        self bleu 计算是将抽取一个 comment 剩余为 reference 句子

        auto_comments: [[comment1][comment2][comment3]]

        return: avg_score vale range [0-100]

        refs_comments 最佳参考是 生成 5条评论，其他4条评论做参考，目前将其他文章的生成评论看做参考

        单位为 %
    """
    from sacrebleu.metrics import BLEU
    Bleu = BLEU(tokenize='zh')

    score_list = []
    num = len(auto_comments) #number of sentences

    for i in range(num):
        refs_comments = [auto_comments[j] for j in range(num) if j!=i]
        auto_comment = auto_comments[i]

        blue = Bleu.corpus_score(auto_comment, refs_comments)   #calcuate score
        score_list.append(blue.score)                           #save the score

        #show each score (Comment out if not needed)
        # print(num, sum(score_list) / num)

    #average score
    avg_score = round(sum(score_list) / num, 2)
    return avg_score


def calc_distinct_n(auto_comments, n_gram=3):
    """ 当前该distinct_n 计算的方式还存在问题，后续再改进, 暂时不使用"""
    from util.distinct_n.metrics import distinct_n_corpus_level
    auto_comment = [[char_zh for char_zh in auto_comment[0]] for auto_comment in auto_comments]
    distinct_socre = distinct_n_corpus_level(auto_comment, n_gram)*100

    return distinct_socre


"""----------------------- test part ------------------------"""


def test_bleu():
    from sacrebleu.metrics import BLEU
    """
    bleu 的计算方式有两种，分别是corpus_score和sentence_score
    corpus_score用于有多个句子，每个句子有多个参考句的时候，sentence_score用于参考只有单个句子

    smooth_method 默认采用exp函数
    若在NLTK版本中，smooth_function的结果不同的nltk版本结果也不同
    """

    auto_comment = ['中国土狗。好养，抵抗力好，顾家，通人性']

    # ref_comments = [['我准备要养一只中华了，狗眼不看人低，很多是人眼看狗低罢了'], ['我天天遛土狗呢！怎么滴怎么滴怎么滴']]
    ref_comments = [['我准备要养一只中华了，狗眼不看人低，很多是人眼看狗低罢了'], ['中国土狗。好养，狗眼不看人低,抵抗力好，顾家，通人性']]
    # ref_comments = [['我天天遛土狗呢！怎么滴怎么滴怎么滴']]

    bleu = BLEU(tokenize='zh')
    print(bleu.corpus_score(auto_comment, ref_comments))
# test_bleu()


def test_self_bleu():
    from sacrebleu.metrics import BLEU

    auto_comments = [['我准备要养一只中华了，狗眼不看人低，很多是人眼看狗低罢了'],
                        ['我天天遛土狗呢！怎么滴怎么滴怎么滴'],
                        ['中国土狗。好养，抵抗力好，顾家，通人性']
                        ]
    score_list = []
    num = len(auto_comments) #number of sentences
    Bleu = BLEU(tokenize='zh')
    for i in range(num):
        refs_comments = [auto_comments[j] for j in range(num) if j!=i]
        auto_comment = auto_comments[i]

        blue = Bleu.corpus_score(auto_comment, refs_comments)   #calcuate score
        score_list.append(blue.score)                           #save the score

        #show each score (Comment out if not needed)
        print(num, sum(score_list) / num)

    #average score
    print(sum(score_list) / num)
# test_self_bleu()



def rouge_rouge_zh():
    """ rouge_zh 对输入的非中文的字符进行了过滤 ，去除非中文字符、字母和数字，因此只有纯中文字符的比较"""
    from rouge_zh import Rouge

    # 输入分词后或按字隔开的中文文本
    auto_comment = ['中 国 土 狗 。 好 养 ， 抵 抗 力 好，顾 家，通 人 性']
    # ref_comments = [['中国土狗。好养，抵抗力好，顾家，通人性'], ['我准备要养一只中华了，狗眼不看人低，很多是人眼看狗低罢了']]
    ref_comments = [['我 准 备 要 养 一 只 中  华了  ，狗 眼 不 看 人 低 ， 很 多 是 人 眼 看 狗 低 罢 了']]
    evaluator  = Rouge(metrics=['rouge-n','rouge-l'],
                        max_n=2,    # n-gram windows size
                        length_limit_type='words',
                        alpha=0.5, # Default F1_score
                        weight_factor=1.2,
                        stemming=True)
    print(evaluator.get_scores(auto_comment, ref_comments))
# rouge_rouge_zh()


def test_rouge_package():
    """ 该库没有非中文字符进行过滤 """
    from rouge import Rouge

    # hypothesis = 'Dan walked to the bakery this morning.'
    # reference = 'Dan went to buy scones earlier this morning.'
    auto_comment = '中 国 土 狗 。 好 养 ， 抵 抗 力 好，顾 家，通 人 性'
    ref_comments = '我 准 备 要 养 一 只 中  华了  ，狗 眼 不 看 人 低 ， 很 多 是 人 眼 看 狗 低 罢 了'

    rouge = Rouge()
    scores = rouge.get_scores(auto_comment, ref_comments)
    # scores = rouge.get_scores(' '.join(list('中国土狗。好养，抵抗力好，顾家，通人性')), ' '.join(list('我天天遛土狗呢！怎么滴怎么滴怎么滴')))
    print(scores)
# test_rouge_package()


def test_sacrerouge_rouge():
    """ sacrerouge测试中文目前存在问题，待以后解决 """
    auto_comment = ['中 国 土 狗 。 好 养 ， 抵 抗 力 好，顾 家，通 人 性']
    # ref_comments = [['中国土狗。好养，抵抗力好，顾家，通人性'], ['我准备要养一只中华了，狗眼不看人低，很多是人眼看狗低罢了']]
    ref_comments = [['我 准 备 要 养 一 只 中  华了  ，狗 眼 不 看 人 低 ， 很 多 是 人 眼 看 狗 低 罢 了']]

    from sacrerouge.metrics import Rouge

    rouge = Rouge(max_ngram=1, compute_rouge_l=True)
    print(rouge.score_multi(auto_comment, ref_comments))
# test_sacrerouge_rouge()


def test_meteor():
    from nltk.translate.meteor_score import meteor_score

    hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
    reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']

    reference3 = ['我', '说', '这', '是', '怎', '么', '回', '事', '来', '明', '天', '要', '放', '假', '了']
    reference2 = ['我', '说', '这', '是', '怎', '么', '回', '事']
    hypothesis2 = ['我', '说', '这', '是', '啥', '呢', '我', '说', '这', '是', '啥', '呢']

    hypothesis = [reference3, reference2]

    # reference3：参考译文
    # hypothesis2：生成的文本
    res = round(meteor_score(hypothesis, hypothesis2), 4)
    print(res)

# test_meteor()


def test_distinct():
    from distinct_n.metrics import distinct_n_corpus_level, distinct_n_sentence_level

    sentence1 = "the the the the the".split()
    sentence2 = "the the the the cat".split()
    sentence3 = "the cat sat on the".split()
    sentences4 = [
        'the cat sat on the mat'.split(),
        'mat the on sat cat the'.split(),
        'i do not know'.split(),
        'Sorry but i do not know'.split(),
    ]

    print(distinct_n_sentence_level(sentence1, 1))
    print(distinct_n_sentence_level(sentence2, 1))
    print(distinct_n_sentence_level(sentence3, 2))

    print(distinct_n_corpus_level(sentences4, 4))

# test_distinct()




