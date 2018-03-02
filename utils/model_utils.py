import tensorflow as tf 
import numpy as np 
import codecs

UNK = '<unk>'
UNK_ID = 1
PAD = '<pad>'
PAD_ID = 0

def read_vocab(vocab_file):
    """
    read vocab from vocab_file, return word2id and id2word
    """
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, 'rb')) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
        
    if vocab[0] != PAD or vocab[1] != UNK:
        print("The first vocab word %s %s is not %s %s" % (vocab[0], vocab[1], PAD, UNK))
        vocab = [PAD, UNK] + vocab
        vocab_size += 2

    word2id = {w:i for i,w in enumerate(vocab)}
    id2word = {i:w for i,w in enumerate(vocab)}
    return word2id, id2word

def get_optimizer(opt):
    if opt == 'adam':
        optfn = tf.train.AdamOptimizer
    elif opt == 'sgd':

        optfn = tf.train.GradientDescentOptimizer
    else:
        assert False
    return optfn