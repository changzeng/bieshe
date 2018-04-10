import time
import logging
import argparse
import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import os, thulac, time
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss


def modify_path(path):
	if path[-1] == "/":
		return path
	else:
		return path+"/"


class LDA:
	def __init__(self, model_path = "model/", corpus_path="corpus/avaliable", topic_num=100, model_time=None):
		self.model_time = str(int(time.time())) if model_time == None else model_time
		self.model_path = modify_path(model_path)
		self.model_save_path = self.model_path + modify_path(self.model_time)
		self.corpus_path = modify_path(corpus_path)
		self.load_stopwords()
		self.useless_files = set([".DS_Store"])
		self.topic_num = topic_num

		self.english_model_path = self.model_save_path + "english_topic_{topic_num}.model".format(topic_num = self.topic_num)
		self.chinese_model_path = self.model_save_path + "chinese_topic_{topic_num}.model".format(topic_num = self.topic_num)

		try:
			os.mkdir("model/"+self.model_time)
		except:
			pass

		self.logger = logging.getLogger("LDA")
		self.logger.setLevel(level = logging.INFO)
		handler = logging.FileHandler("log.txt")
		handler.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		self.logger.addHandler(handler)

	def load_stopwords(self):
		with open("stopwords.txt", encoding="utf-8") as fd:
			txt = fd.read()
			self.stopwords = set(txt.split("\n"))

	def is_model_exists(self):
		return os.path.exists(self.model_save_path)

	def load_corpus(self):
		files = list(map(lambda x:self.corpus_path+x, os.listdir(self.corpus_path)))

		corpus = []

		for _file in files:
			with open(_file, encoding="utf-8", errors="ignore") as fd:
				txt = fd.read()
				docs = txt.split("\n\n")[:-1]
				corpus += map(lambda x:x.split("\n"), docs)

		return zip(*corpus)

	def load_model(self):
		self.english_model = LdaModel.load(self.english_model_path)
		self.chinese_model = LdaModel.load(self.chinese_model_path)

	def split(self, text):
		return text.split()

	def prepare_train(self):
		self.logger.info("loading corpus...")
		corpus = list(self.load_corpus())

		english = corpus[0]
		chinese = corpus[1]

		thu = thulac.thulac(seg_only=True)

		self.logger.info("cutting english sentences...")
		english = map(self.split, english)

		self.logger.info("cutting chinese sentences...")
		chinese_cut = map(thu.cut, chinese)
		chinese = []
		for i,item in enumerate(chinese_cut):
			chinese.append(list(map(lambda x:x[0], item)))
		chinese = [[word for word in sent if word not in self.stopwords] for sent in chinese]

		self.logger.info("making dictionaries...")
		self.english_dictionary = Dictionary(english)
		self.chinese_dictionary = Dictionary(chinese)

		self.logger.info("translating words to ids...")
		self.english_ids = [ self.english_dictionary.doc2bow(text) for text in english ]
		self.chinese_ids = [ self.chinese_dictionary.doc2bow(text) for text in chinese ]

	def train(self, topic_num=None):
		self.topic_num = self.topic_num if topic_num is None else topic_num

		self.english_model_path = self.model_save_path + "english_topic_{topic_num}.model".format(topic_num = self.topic_num)
		self.chinese_model_path = self.model_save_path + "chinese_topic_{topic_num}.model".format(topic_num = self.topic_num)

		self.logger.info("training english model....,topic num: %3d" % (topic_num))
		self.english_model = LdaModel(corpus=self.english_ids, id2word=self.english_dictionary, num_topics=topic_num)
		self.logger.info("training chinese model....,topic num: %3d" % (topic_num))
		self.chinese_model = LdaModel(corpus=self.chinese_ids, id2word=self.chinese_dictionary, num_topics=topic_num)
		self.logger.info("model training complete")

		self.save()

	def save(self):
		self.english_model.save(self.english_model_path)
		self.chinese_model.save(self.chinese_model_path)

	def predict_english(self):
		pass

	def predict_chinese(self):
		pass


class LDA_LSTM:
	def __init__(self, batch_size=50, topic_num=30, hidden_size=100, corpus_path="corpus/avaliable"):
		self.batch_size = batch_size
		self.topic_num = topic_num
		self.hidden_size = hidden_size
		self.corpus_path = modify_path(corpus_path)
		self.save_step = 30
		self.thu = thulac.thulac(seg_only=True)
		self.model_path = "./model/LSTM/topic_num(%d)-hidden_size(%d)-batch_size(%d)/" % (self.topic_num, self.hidden_size, self.batch_size)
		if not os.path.exists(self.model_path):
			os.mkdir(self.model_path)
		self.load_stopwords()

		self.load_lda()
		self.build_graph()
		self.saver = tf.train.Saver() 

	def build_graph(self):
		english_input = tf.placeholder(tf.float32, [None, self.topic_num, self.topic_num], name="english_input")
		chinese_input = tf.placeholder(tf.float32, [None, self.topic_num, self.topic_num], name="chinese_input")
		
		Y = tf.placeholder(tf.float32, [None], name="scores")

		two_lstm_outputs = []

		for i in range(2):
			with tf.variable_scope("lstm-%s" % chr(ord('a') + i)):
				if i == 0:
					X = english_input
				else:
					X = chinese_input

				cell = BasicLSTMCell(num_units=self.hidden_size, name="basic-lstm-cell")
				initial_state = cell.zero_state(self.batch_size, tf.float32)
				outputs, _states = tf.nn.dynamic_rnn(cell,
				                                     X,
				                                     initial_state=initial_state,
				                                     dtype=tf.float32)

				outputs = tf.slice(outputs, [0,self.topic_num-1,0], [self.batch_size, 1, self.hidden_size])
				two_lstm_outputs.append(tf.reshape(outputs, [-1, self.hidden_size]))

		concat_outputs = tf.concat(two_lstm_outputs, 1)

		y = fully_connected(inputs=concat_outputs,
				                          num_outputs=1,
				                          activation_fn=None)
		y = tf.reshape(y, [self.batch_size])
		y = tf.nn.sigmoid(y)

		self.global_step = tf.Variable(0, trainable=False)  
		self.learning_rate = tf.train.exponential_decay(0.1, self.global_step, 10, 2, staircase=False)  
		self.loss_op = tf.reduce_mean(tf.square(y - Y))
		self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss_op, global_step=self.global_step)

		self.prediction = y

	def load_stopwords(self):
		with open("stopwords.txt", encoding="utf-8") as fd:
			txt = fd.read()
			self.stopwords = set(txt.split("\n"))

	def gen_feed_dict(self, english_batch, chinese_batch, scores):
		return {"english_input:0": english_batch, "chinese_input:0": chinese_batch, "scores:0": scores}

	def train(self):
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)
			for english_batch, chinese_batch in self.gen_batch():
				feed_dict = self.gen_feed_dict(english_batch, chinese_batch, np.ones(shape=(self.batch_size,)))
				loss, _, prediction, global_step = sess.run([self.loss_op, self.train_op, self.prediction, self.global_step], feed_dict=feed_dict)
				print(prediction)
				if global_step % self.save_step == 0:
					print("saving model....")
					self.saver.save(sess, self.model_path+"model.ckpt")

	def txt_2_fea(self, english_batch, chinese_batch):
		english_batch = [english.split() for english in english_batch]
		chinese_batch = [[word[0] for word in self.thu.cut(chinese) if word[0] not in self.stopwords] for chinese in chinese_batch]
		
		english_bow_batch = [self.english_lda.id2word.doc2bow(english_words) for english_words in english_batch]
		chinese_bow_batch = [self.chinese_lda.id2word.doc2bow(chinese_words) for chinese_words in chinese_batch]

		english_topics_batch = [self.english_lda[english_bow] for english_bow in english_bow_batch]
		chinese_topics_batch = [self.chinese_lda[chinese_bow] for chinese_bow in chinese_bow_batch]

		english_batch = []
		chinese_batch = []

		for i in range(len(english_topics_batch)):
			english_input = np.zeros(shape=(self.topic_num, self.topic_num), dtype=np.float32)
			chinese_input = np.zeros(shape=(self.topic_num, self.topic_num), dtype=np.float32)

			english_topics = english_topics_batch[i]
			chinese_topics = chinese_topics_batch[i]

			for item in english_topics:
				index = item[0]
				score = item[1]
				english_input[index][index] = score

			for item in chinese_topics:
				index = item[0]
				score = item[1]
				chinese_input[index][index] = score
			english_batch.append(english_input)
			chinese_batch.append(chinese_input)

		return english_batch, chinese_batch

	def gen_batch(self):
		files = list(map(lambda x:self.corpus_path+x, os.listdir(self.corpus_path)))

		batch_num = 0
		english_batch = []
		chinese_batch = []
		for _file in files:
			with open(_file, encoding="utf-8", errors="ignore") as fd:
				while True:
					batch_num += 1

					english = fd.readline().strip()
					chinese = fd.readline().strip()
					fd.readline()
					if len(english) == 0 or len(chinese) == 0:
						break

					english_batch.append(english)
					chinese_batch.append(chinese)

					# saving model
					if batch_num % self.batch_size == 0:
						yield self.txt_2_fea(english_batch, chinese_batch)
						english_batch = []
						chinese_batch = []
						batch_num = 0

	def load_lda(self):
		lda = LDA(model_time="1515053496", topic_num=self.topic_num)
		lda.load_model()

		self.english_lda = lda.english_model
		self.chinese_lda = lda.chinese_model
		scores = 1

	def predict_raw(self, english_batch, chinese_batch):
		english_batch, chinese_batch = self.txt_2_fea(english_batch, chinese_batch)
		with tf.Session() as sess:
			sess.run([self.predict], feed_dict={"english_input:0": english_batch, "chinese_input0": chinese_batch})


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='bieshe artwork.')
	parser.add_argument('--mode', default="lda", help='train lda model')
	args = parser.parse_args()
	if args.mode == "train_lda":
		lda = LDA(model_time="1515053496")
		lda.prepare_train()
		for topic_num in range(10, 100):
			lda.train(topic_num = topic_num)
	elif args.mode == "train_lstm":
		lda_lstm = LDA_LSTM()
		lda_lstm.train()
		# for batch in lda_lstm.gen_batch():
		# 	print(batch)
