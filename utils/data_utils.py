import urllib.request
import zipfile
import lxml.etree
import os
from collections import Counter
import codecs
import math
import random
import re

from utils.model_utils import UNK, UNK_ID, PAD, PAD_ID

from sklearn.model_selection import train_test_split

def download_data():
    #TODO: if line doesn't work
    if not os.path.isfile('/home/zhuyuhe/mydata/oxford-cs-deepnlp-2017/practical-2/model/data/ted_en-20160408.zip'):
        print("downloading data...")
        urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="data/ted_en-20160408.zip")


def load_data_from_zip():
    with zipfile.ZipFile('data/ted_en-20160408.zip', 'r') as z:
        doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

    input_text = doc.xpath('//content/text()')
    input_text_keywords = doc.xpath('//keywords/text()') 
    input_text_title = doc.xpath('//title/text()')
    input_text_summary = doc.xpath('//description/text()')
    return (input_text, input_text_keywords, input_text_title, input_text_summary)   

def filter_keywords(x):
    x = x.lower()
    res = ''
    if 'technology' in x:
        res += 'T'
    else:
        res += 'o'
    if 'entertainment' in x:
        res += 'E'
    else:
        res += 'o'
    if 'design' in x:
        res += 'D'
    else:
        res += 'o'
    return res

def tcdata_process(input_text, input_text_tag):
    """
    data preprocess for text classification
    train/test data split, data save
    
    input_text: TED talks text
    input_text_tag: text label
        – None of the keywords → ooo
        – “Technology” → Too
        – “Entertainment” → oEo
        – “Design” → ooD
        – “Technology” and “Entertainment” → TEo
        – “Technology” and “Design” → ToD
        – “Entertainment” and “Design” → oED
        – “Technology” and “Entertainment” and “Design” → TED
    """
    #data process
    input_text = [re.sub(r'\([^)]*\)', '', x).lower() for x in input_text]
    input_text = [re.sub(u'[^0-9a-zA-Z \n]', '', x) for x in input_text]
    # input_text = [x.split() for x in input_text]
    input_text_tag = [filter_keywords(t) for t in input_text_tag]
    input_text_pairs = list(zip(input_text, input_text_tag))

    data_size = len(input_text_pairs)

    train_data, test_data = train_test_split(input_text_pairs, test_size = 0.1, train_size = 0.9, random_state = 5)
    save_data(train_data, 'train.data')
    save_data(test_data, 'test.data')

    return train_data, test_data
        

def save_data(data, fname):
    with codecs.open('data/' + fname,'w', 'utf-8') as file:
        for pairs in data:
            file.write(pairs[0].replace('\n', ' '))
            file.write('\t')
            file.write(pairs[1])
            file.write('\n')
    

def create_vocab(data, lower_case = False, min_cnt = 0):
    print("Create vocab with lower case: {0}, min count: {1}".format(lower_case, min_cnt))
    word_count = Counter()
    tag_count = Counter()
    texts, tags = zip(*data)
    texts = [text.split() for text in texts]
    tag_count.update(tags)
    for text in texts:
        word_count.update([t.lower() if lower_case else t for t in text])
    word_vocab = [PAD,UNK]
    tag_vocab = []
    for w,c in word_count.most_common():
        if c < min_cnt:
            break
        word_vocab.append(w)
    for t,c in tag_count.most_common():
        tag_vocab.append(t)
    print("word vocab size: {0}, tag vocab size: {1}".format(len(word_vocab), len(tag_vocab)))
    return word_vocab, tag_vocab

def save_vocab(vocab, fname):
    with codecs.open(fname, 'w', 'utf-8') as f:
        for w in vocab:
            f.write(w + '\n')

def convert_dataset(data, word_vocab, tag_vocab):
    word2id = {w:i for i,w in enumerate(word_vocab)}
    tag2id = {t:i for i,t in enumerate(tag_vocab)}
    res = []
    for pairs in data:
        words, tag = pairs
        words = [word2id[w] if w in word2id else word2id[UNK] for w in words]
        tag = tag2id[tag]
        res.append((words, len(words), tag))
    return res


class Batch(object):
    def __init__(self, data, batch_size = 20):
        self.data_size = len(data)
        self.batch_size = batch_size
        self.num_batch = int(math.ceil(self.data_size / self.batch_size))
    
        self.data = sorted(data, key = lambda x: x[1])
        self.batch_data = self.patch2batches()
        

    def patch2batches(self):
        batch_data = list()
        for i in range(self.num_batch):
            batch_data.append(self.pad_data(self.data[i*self.batch_size : (i+1)*self.batch_size]))
        return batch_data

    def pad_data(self, batch_data):
        #每个batch的数据维度需要一致,对于较短的句子需要做填充处理
        max_length = max([data[1] for data in batch_data])
        padded_data = []
        for data in batch_data:
            words, length, tag = data
            padding = [PAD_ID] * (max_length - length)
            padded_data.append((words + padding, length, tag))
        return padded_data

    def next_batch(self, shuffle = True):
        if shuffle:
            random.shuffle(self.batch_data)
        for batch in self.batch_data:
            yield batch





