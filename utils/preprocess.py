import argparse
import json

import utils


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default='config_tencent.yaml', type=str,
                        help="config file")
    return parser.parse_args()


def build_tencent_vocab(corpus_files, vocab_file):
    word2count = {}
    for corpus_file in corpus_files:
        for line in open(corpus_file):
            # words = line.strip().split()
            g = json.loads(line)
            words = g["body"].split()
            words.extend(g["title"].split())
            words.extend([w for com in g["comment"] for w in com[0].split()])
            for word in words:
                if word not in word2count:
                    word2count[word] = 0
                word2count[word] += 1
    word2count = list(word2count.items())
    word2count.sort(key=lambda k: k[1], reverse=True)
    write = open(vocab_file, 'w')
    for word_pair in word2count:
        write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
    write.close()


def build_yahoo_vocab(corpus_files, vocab_file):
    word2count = {}
    for corpus_file in corpus_files:
        for line in open(corpus_file):
            # words = line.strip().split()
            g = json.loads(line)
            words = ' '.join(g['paras']).split()
            words.extend(g["title"].split())
            words.extend([w for com in g["cmts"] for w in com['cmt'].split()])
            for word in words:
                if word not in word2count:
                    word2count[word] = 0
                word2count[word] += 1
    word2count = list(word2count.items())
    word2count.sort(key=lambda k: k[1], reverse=True)
    write = open(vocab_file, 'w')
    for word_pair in word2count:
        write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
    write.close()


def build_163_vocab(corpus_files, vocab_file):
    word2count = {}
    for corpus_file in corpus_files:
        for line in open(corpus_file):
            # words = line.strip().split()
            g = json.loads(line)
            words = g["body"].split()
            words.extend(g["title"].split())
            words.extend([w for com in g["cmts"] for w in com['cmt'].split()])
            for word in words:
                if word not in word2count:
                    word2count[word] = 0
                word2count[word] += 1
    word2count = list(word2count.items())
    word2count.sort(key=lambda k: k[1], reverse=True)
    write = open(vocab_file, 'w')
    for word_pair in word2count:
        write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
    write.close()


if __name__ == "__main__":
    args = parse_config()
    config = utils.util.read_config(args.config)
    corpus_files = [config.train_file, config.valid_file, config.test_file]
    vocab_file = config.vocab_file

    if 'tencent' in args.config:
        build_tencent_vocab(corpus_files, vocab_file)
    elif 'yahoo' in args.config:
        build_yahoo_vocab(corpus_files, vocab_file)
    elif '163' in args.config:
        build_163_vocab(corpus_files, vocab_file)
