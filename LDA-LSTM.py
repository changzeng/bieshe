import time
import shutil
import logging
import argparse
import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import os, thulac, time
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.seq2seq import sequence_loss


def modify_path(path):
	if path[-1] == "/":
		return path
	else:
		return path+"/"


class LDA:
	def __init__(self, model_path = "model/LDA/", corpus_path="corpus/avaliable", topic_num=100, model_time=None):
		self.model_save_path = self.get_model_save_path(topic_num)
		self.corpus_path = modify_path(corpus_path)
		self.load_stopwords()
		self.useless_files = set([".DS_Store"])
		self.topic_num = topic_num

		self.english_model_path = self.model_save_path + "english_topic_{topic_num}.model".format(topic_num = self.topic_num)
		self.chinese_model_path = self.model_save_path + "chinese_topic_{topic_num}.model".format(topic_num = self.topic_num)

		try:
			os.mkdir(self.model_save_path)
		except:
			pass

		self.logger = logging.getLogger("LDA")
		self.logger.setLevel(level = logging.INFO)
		handler = logging.FileHandler("log.txt")
		handler.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		self.logger.addHandler(handler)

	def get_model_save_path(self, topic_num):
		path = "model/LDA/" + "topic_num({topic_num})/".format(topic_num = topic_num)
		return modify_path(path)

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
		
		english, chinese = zip(*corpus)
		english = [sentence.strip() for sentence in english]
		chinese = [sentence.strip() for sentence in chinese]

		return english, chinese

	def load_model(self):
		self.english_model = LdaModel.load(self.english_model_path)
		self.chinese_model = LdaModel.load(self.chinese_model_path)

	def split(self, text):
		return text.split()

	def prepare_train(self):
		self.logger.info("loading corpus...")
		english, chinese = list(self.load_corpus())

		self.logger.info("cutting english sentences...")
		cut_result = []
		for sentence in english:
			cut_result.append([word for word in sentence.split()])
		english = cut_result

		self.logger.info("cutting chinese sentences...")
		cut_result = []
		for i, sentence in enumerate(chinese):
			cut_result.append(sentence.split(" "))
		chinese = cut_result

		self.logger.info("making dictionaries...")
		self.english_dictionary = Dictionary(english)
		self.chinese_dictionary = Dictionary(chinese)

		self.logger.info("translating words to ids...")
		self.english_ids = [ self.english_dictionary.doc2bow(text) for text in english ]
		self.chinese_ids = [ self.chinese_dictionary.doc2bow(text) for text in chinese ]

	def train(self, topic_num=None):
		self.topic_num = self.topic_num if topic_num is None else topic_num

		model_path = self.get_model_save_path(topic_num)
		if not os.path.exists(model_path):
			os.mkdir(model_path)

		self.english_model_path = model_path + "english_topic_{topic_num}.model".format(topic_num = self.topic_num)
		self.chinese_model_path = model_path + "chinese_topic_{topic_num}.model".format(topic_num = self.topic_num)

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
	def __init__(self, batch_size=50, topic_num=30, hidden_size=100, corpus_path="corpus/avaliable", debug=True):
		self.batch_size = batch_size
		self.topic_num = topic_num
		self.hidden_size = hidden_size
		self.corpus_path = modify_path(corpus_path)
		self.save_step = 30
		self.thu = thulac.thulac(seg_only=True)
		self.model_path = "./model/LSTM/topic_num(%d)-hidden_size(%d)-batch_size(%d)/" % (self.topic_num, self.hidden_size, self.batch_size)
		self.model_save_path = self.model_path + "model/"
		self.summary_save_path = self.model_path + "summary/"

		if debug == True:
			if os.path.exists(self.model_path):
				shutil.rmtree(self.model_path)

		# creat directory
		for _dir in [self.model_path, self.model_save_path, self.summary_save_path]:
			if not os.path.exists(_dir):
				os.mkdir(_dir)

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

				cell = BasicLSTMCell(num_units=self.hidden_size)
				initial_state = cell.zero_state(self.batch_size, tf.float32)
				outputs, _states = tf.nn.dynamic_rnn(cell,
				                                     X,
				                                     initial_state=initial_state,
				                                     dtype=tf.float32)

				outputs = tf.slice(outputs, [0,self.topic_num-1,0], [self.batch_size, 1, self.hidden_size])
				two_lstm_outputs.append(tf.reshape(outputs, [-1, self.hidden_size]))

		# concat and reshape output
		concat_outputs = tf.concat(two_lstm_outputs, 1)
		concat_outputs = tf.reshape(concat_outputs, [-1, 2*self.hidden_size])

		# full connected layer
		w = tf.Variable(tf.random_normal([2*self.hidden_size, 1]), name="weight", dtype=tf.float32)
		b = tf.Variable(tf.constant(1.0), name="bias", dtype=tf.float32)
		y = tf.matmul(concat_outputs, w) + b
		y = tf.nn.sigmoid(y)

		self.global_step = tf.Variable(0, trainable=False)  
		self.learning_rate = tf.train.exponential_decay(0.1, self.global_step, 10, 2, staircase=False)  
		self.loss_op = tf.reduce_mean(tf.square(y - Y))
		self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss_op, global_step=self.global_step)

		tf.summary.scalar("loss", self.loss_op)

		self.prediction = y

		# 导出图
		# print("exporting meta graph......")
		# tf.train.export_meta_graph(filename=self.model_path+"model.ckpt.meta") 

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
			merged = tf.summary.merge_all()
			file_writer = tf.summary.FileWriter(self.summary_save_path, sess.graph)
			for (english_batch, chinese_batch), labels in self.gen_batch():
				feed_dict = self.gen_feed_dict(english_batch, chinese_batch, labels)
				loss, _, summary, prediction, global_step = sess.run([self.loss_op, self.train_op, merged, self.prediction, self.global_step], feed_dict=feed_dict)
				step = global_step % self.save_step
				print("current_step/save_step: (%4d)/(%4d)" % (self.save_step if step == 0 else step, self.save_step))
				if global_step % self.save_step == 0:
					print("saving model....")
					self.saver.save(sess, self.model_save_path+"model.ckpt")
				# write summary
				file_writer.add_summary(summary, global_step)

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

	def gen_batch(self, pos_proportion=0.5):
		pos_file = self.corpus_path+"pos.txt"
		neg_file = self.corpus_path+"neg.txt"

		pos_num = int(self.batch_size * pos_proportion)
		neg_num = self.batch_size - pos_num

		batch_num = 0
		english_batch = []
		chinese_batch = []
		score_list = []

		with open(pos_file, encoding="utf-8", errors="ignore") as pos_fd:
			with open(neg_file, encoding="utf-8", errors="ignore") as neg_fd:
				while True:
					for _ in range(pos_num):
						english = pos_fd.readline().strip()
						chinese = pos_fd.readline().strip()
						pos_fd.readline()
						if len(english) == 0 or len(chinese) == 0:
							break

						english_batch.append(english)
						chinese_batch.append(chinese)
						score_list.append(1)

					for _ in range(neg_num):
						english = neg_fd.readline().strip()
						chinese = neg_fd.readline().strip()
						neg_fd.readline()
						if len(english) == 0 or len(chinese) == 0:
							break

						english_batch.append(english)
						chinese_batch.append(chinese)
						score_list.append(0)

					# saving model
					if batch_num % self.batch_size == 0:
						yield self.txt_2_fea(english_batch, chinese_batch), np.array(score_list, dtype=np.float32)
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
	parser.add_argument('--mode', type=str, default="lda", help='train lda model')
	parser.add_argument('--topic_num', type=int, default=100, help='lda topic num')
	parser.add_argument('--batch_size', type=int, default=50, help='batch size')
	parser.add_argument('--debug', type=bool, default=False, help='debug mode')
	args = parser.parse_args()
	if args.mode == "train_lda":
		lda = LDA(model_time="1515053496")
		lda.prepare_train()
		if args.topic_num is not None:
			lda.train(topic_num = int(args.topic_num))
		else:
			for topic_num in range(10, 100):
				lda.train(topic_num = topic_num)
	elif args.mode == "train_lstm":
		lda_lstm = LDA_LSTM(batch_size=args.batch_size, topic_num=args.topic_num, debug=args.debug)
		lda_lstm.train()
		# for batch in lda_lstm.gen_batch():
		# 	print(batch)
	elif args.mode == "test":
		lda_lstm = LDA_LSTM()
		num = 0
		for batch in lda_lstm.gen_batch():
			print(batch)
			num += 1
			if num >= 1:
				break
