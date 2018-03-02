class Config(object):
    def __init__(self):
        self.checkpoint_dir = '/data/home/zhuyuhe/mydata/oxford-cs-deepnlp-2017/practical-2/model/checkpoint/'
        self.mode = 'train'
        self.num_gpus = 1
        
        #traning
        self.learning_rate = 0.01
        self.optimizer = 'adam'
        self.max_gradient_norm = 5.0
        self.keep_prob = 1.0
        self.num_train_steps = 200000
        self.learning_decay = False
        self.leraning_decay_steps = None
        self.decay_times = 10
        self.decay_factor = 0.98
        self.batch_size = 32

        #embedding
        self.vocab_file = 'vocab.txt'
        self.pretrained_embedding_file = None
        self.embedding_size = 50

        #network
        self.num_units = 50
        self.epoches = 10

        #labels
        self.num_tags = 8
        self.tag_vocab_file = None

