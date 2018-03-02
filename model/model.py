import tensorflow as tf 
import os

from utils.model_utils import read_vocab, get_optimizer

class DenseModel(object):
    def __init__(self, sess, config):
        assert config.mode in ['train', 'eval']
        self.config = config
        self.sess = sess
        self.train_phase = config.mode == 'train'

    def build(self):
        print("building graph...")
        self.global_step = tf.Variable(0, trainable = False)
        self.setup_input_placeholders()
        self.setup_embeddings()
        self.setup_projection()
        if self.train_phase:
            self.setup_train()

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def setup_input_placeholders(self):
        #batch_size*sentence_length
        self.source_tokens = tf.placeholder(tf.int32, shape = [None, None], name = 'source_tokens')
        #batch_size * embedding_size
        self.source_length = tf.placeholder(tf.int32, shape = [None, None], name = 'source_length')
        #batch_size
        #根据self.target_tokens 计算self.target_tag: batch_size * num_tags (类别标签的one-hot编码)
        self.target_tokens = tf.placeholder(tf.int32, shape = [None], name = 'target_tag')

        if self.train_phase:
            self.keep_prob = tf.placeholder(tf.float32, name = 'Dropout')
        
        self.batch_size = tf.shape(self.source_tokens)[0]

    def setup_embeddings(self):
        with tf.variable_scope('Embedding'), tf.device("/cpu:0"):
            self.word2id, self.id2word = read_vocab(self.config.vocab_file)
            if self.config.pretrained_embedding_file:
                embedding = load_pretrained_emb_from_txt(self.id2word, self.config.pretrain_embedding_file)
                self.source_embedding = tf.get_variable("source_embedding", dtype = tf.float32, initializer = tf.constant(embedding))
            else:
                self.source_embedding = tf.get_variable("source_embedding", [len(self.id2word), self.config.embedding_size], dtype = tf.float32, initializer = tf.random_uniform_initializer(-1,1))
            
            #batch_size*sentence_length(max)*embedding_size
            self.source_inputs = tf.nn.embedding_lookup(self.source_embedding, self.source_tokens)

            if self.train_phase:
                self.source_inputs = tf.nn.dropout(self.source_inputs, self.keep_prob)
    
    def setup_projection(self):
        with tf.variable_scope('Projection'):
            #X = (1/N)(\sum_xi)
            #batch_size * embedding_size
            self.input_X = tf.reduce_mean(self.source_inputs, axis = 1, keep_dims = False)
            # self.input_X = tf.div(tf.reduce_sum(self.source_inputs, axis = 1), tf.cast(self.source_length, tf.float32))
            # hidden_layer = tf.layers.Dense(self.config.num_units, tf.tanh, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1), name = 'hidden_layer')
            # #batch_size * self.num_units
            # hidden_value = hidden_layer(self.input_X)
            W1 = tf.Variable(tf.random_normal((self.config.embedding_size, self.config.num_units), stddev=0.1), name="W1")
            b1 = tf.Variable(tf.zeros(self.config.num_units), name='b1')

            W2 = tf.Variable(tf.random_normal((self.config.num_units, self.config.num_tags), stddev=0.1), name="W2")
            b2 = tf.Variable(tf.zeros(self.config.num_tags), name='b2')

            hidden_value = tf.nn.tanh(tf.matmul(tf.cast(self.input_X, tf.float32), W1) + b1)

            # hidden_layer2 = tf.layers.Dense(self.config.num_units / 2, tf.tanh, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1), name = 'hidden_layer2')
            # hidden_value2 = hidden_layer2(hidden_value1)

            if self.train_phase:
                hidden_value = tf.nn.dropout(hidden_value, self.keep_prob)
                # hidden_value2 = tf.nn.dropout(hidden_value2, self.keep_prob)
            # project_layer = tf.layers.Dense(self.config.num_tags, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1), name = 'project_layer')
            #batch_size * self.num_tags

            # output = tf.matmul(hidden_value, W2) + b2
            # self.pred = project_layer(hidden_value)
            self.pred = tf.matmul(hidden_value, W2) + b2
            #batch_size * self.num_tags
            with tf.name_scope("loss"):
                self.target_tag = tf.one_hot(self.target_tokens, depth = self.config.num_tags, on_value = 1.0, off_value = 0.0)
                self.pred_pro = tf.nn.softmax(self.pred)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_tag, logits = self.pred))
            with tf.name_scope("accuracy"):
                self.predictions = tf.argmax(self.pred, axis = 1)
                self.correct_pred = tf.equal(self.predictions, tf.argmax(self.target_tag, axis = -1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def setup_train(self):
        print("set up training...")
        self.learning_rate = tf.constant(self.config.learning_rate)
        opt = get_optimizer(self.config.optimizer)(self.learning_rate)
        self.train = opt.minimize(self.loss)
        # params = tf.trainable_variables()
        # gradients = tf.gradients(self.loss, params)


    def predict(self, source_tokens, source_length):
        length = []
        for l in source_length:
            length.append([l] * self.config.embedding_size)
        feed_dict = {
            self.source_tokens: source_tokens,
            self.source_length: length
        }
        if self.train_phase:
            feed_dict[self.keep_prob] = 1.0

        predictions = self.sess.run([self.predictions], feed_dict = feed_dict)
        return predictions


    def train_one_batch(self, source_tokens, source_length, target_tag):
        length = []
        for l in source_length:
            length.append([l] * self.config.embedding_size)
        feed_dict = {
            self.source_tokens: source_tokens,
            self.source_length: length,
            self.target_tokens: target_tag,
            self.keep_prob: self.config.keep_prob
        }
        accuracy, loss, _ = self.sess.run([self.accuracy, self.loss, self.train], feed_dict = feed_dict)
        return accuracy, loss

class RNNModel(DenseModel):
    def __init__(self, sess, config):
        pass