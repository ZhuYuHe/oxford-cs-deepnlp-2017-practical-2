import os
import pickle
import time
import sys

import numpy as np

import tensorflow as tf 

from utils.data_utils import Batch, convert_dataset, create_vocab,\
download_data, load_data_from_zip, tcdata_process, save_vocab
from model.config import Config
from model.model import DenseModel
from utils.train_utils import get_config_proto

if __name__ == '__main__':
    DATA_DIR = './data'
    checkpoint_dir = '/home/zhuyuhe/mydata/oxford-cs-deepnlp-2017/practical-2/checkpoint/'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    #read data
    download_data()
    input_text, input_text_tag, _, _ = load_data_from_zip()
    train_data, test_data = tcdata_process(input_text, input_text_tag)
    print("training data size: {0}, test data size: {1}".format(len(train_data), len(test_data)))

    #create vocab from training data
    word_vocab, tag_vocab = create_vocab(train_data, lower_case = True, min_cnt = 2)

    #save vocab
    save_vocab(word_vocab, os.path.join(checkpoint_dir, "word.vocab"))
    save_vocab(tag_vocab, os.path.join(checkpoint_dir, "tag.vocab"))

    #convert word into ids
    train_data = convert_dataset(train_data, word_vocab, tag_vocab)
    sys.exit()
    print("training data size: {0}".format(len(train_data)))
    train_data_batch = Batch(train_data, 50)

    test_data = convert_dataset(test_data, word_vocab, tag_vocab)
    print("test data size: {0}".format(len(test_data)))
    test_data_batch = Batch(test_data, 200)

    #create model
    config = Config()

    # #update config
    # if os.path.exists(os.path.join(checkpoint_dir, 'config.pkl')):
    #     config = pickle.load(open(os.path.join(checkpoint_dir, 'config.pkl'),'rb'))
    # else:
    config.checkpoint_dir = checkpoint_dir
    config.vocab_file = os.path.join(checkpoint_dir, 'word.vocab')
    config.num_tag = len(tag_vocab)
    config.tag_vocab_file = os.path.join(checkpoint_dir, 'tag.vocab')
    # pickle.dump(config, open(os.path.join(config.checkpoint_dir, 'config.pkl'), 'wb'))

    with tf.Session(config = get_config_proto(log_device_placement=False)) as sess:
        model = DenseModel(sess, config)
        #model = RNNModel()
        model.build()
        model.init()

        print("Config:")
        for k,v in config.__dict__.items():
            print(k, '-', v, sep = '\t')
        
        for epoch in range(config.epoches):
            i = 0
            epoch_train_acc, epoch_loss = [], []
            for batch in train_data_batch.next_batch():
                accuracy, loss = model.train_one_batch(*zip(*batch))
                epoch_train_acc.append(accuracy)
                epoch_loss.append(loss)
            print("epoch: {0}, epoch train loss: {1}, epoch train accuracy: {2}".format(epoch, np.mean(epoch_loss), np.mean(epoch_train_acc)))



        

