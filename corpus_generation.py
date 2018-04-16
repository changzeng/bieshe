# encoding: utf-8

import os
import thulac
import argparse
from random import randint, shuffle
from sys import getsizeof


class Generator(object):
    def __init__(self):
        # training dataset
        self.en_train_not_cut = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.en"
        self.zh_train_not_cut = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.zh"
        self.en_train = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.en.cut"
        self.zh_train = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.zh.cut"
        # test dataset
        self.en_test_not_cut = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.en.test"
        self.zh_test_not_cut = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.zh.test"
        self.en_test = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.en.test.cut"
        self.zh_test = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.zh.test.cut"
        # key parameters
        self.max_corpus_size = 50 * 1024 * 1024
        self.neg_num = 10
        self.min_line_num = 100
        self.max_line_num = 150
        self.line_step = 2

        self.load_stopwords()

    def cut(self, en_file, zh_file):
        if os.path.exists(en_file+".cut") and os.path.exists(zh_file+".cut"):
            return
        # 此处应添加删除多余文件代码
        # ---------------------
        def cut_raw(file_name, cut_func, max_memory_size = 30*1024*1024):
            item_list = []
            item_list_size = 0
            with open(file_name, encoding="utf-8", errors="ignore") as fd:
                for line in fd.readlines():
                    cut_result = [word for word in cut_func(line.strip()) if word not in self.stopwords]
                    item_str = " ".join(cut_result)
                    item_list.append(item_str)
                    item_list_size += len(item_str)
                    if item_list_size >= max_memory_size:
                        with open(file_name+".cut", "a", encoding="utf-8") as fd:
                            fd.write("\n".join(item_list))
                        item_list = []
                        item_list_size = 0

            with open(file_name+".cut", "a", encoding="utf-8") as fd:
                fd.write("\n".join(item_list))

        def en_word_filter(en_word):
            if len(en_word) == 0:
                return ""
            # 将单词转化为小写
            en_word = en_word.lower()
            # 除去英文停用词
            if en_word in self.stopwords_english:
                return ""
            # 除去标点符号
            if en_word in self.stopwords_punctuation:
                return ""
            # 去掉单词首尾的标点符号
            while en_word[0] in self.stopwords_punctuation:
                if len(en_word) == 1:
                    return ""
                en_word = en_word[1:]
            while en_word[-1] in self.stopwords_punctuation:
                if len(en_word) == 1:
                    return ""
                en_word = en_word[:-1]
            return en_word

        def split_en(_str):
            result = []
            # 将转义字符变为空格
            blank_words = ["&nbsp", "-", "–"]
            for word in blank_words:
                _str = _str.replace(word, " ")
            # 去除特殊字符
            remove_words = ["’s"]
            for word in remove_words:
                _str = _str.replace(word, "")
            for en_word in _str.split(" "):
                fileter_en_word = en_word_filter(en_word)
                if len(fileter_en_word) > 0:
                    result.append(fileter_en_word)
            return result

        thu = thulac.thulac(seg_only=True)
        def split_zh(_str):
            return [cut_item[0] for cut_item in thu.cut(_str)]

        cut_raw(en_file, split_en)
        cut_raw(zh_file, split_zh)

    def load_stopwords(self):
        with open("stopwords.txt", encoding="utf-8") as fd:
            txt = fd.read()
            self.stopwords = set(txt.split("\n"))

        with open("stopwords_english.txt", encoding="utf-8") as fd:
            txt = fd.read()
            self.stopwords_english = set(txt.split("\n"))

        with open("stopwords_punctuation.txt", encoding="utf-8") as fd:
            txt = fd.read()
            self.stopwords_punctuation = set(txt.split("\n"))

    def load_file(self, en_file, zh_file):
        def load(file_name):
            with open(file_name, encoding="utf-8", errors="ignore") as fd:
                lines = fd.read().strip().split("\n")
                lines = [line.strip() for line in lines]
                return lines
        self.en_lines = load(en_file)
        self.zh_lines = load(zh_file)

    def random(self, _range, _num, block_list=set()):
        num_set = set()
        while len(num_set) < _num:
            num_set.add(randint(_range[0], _range[1]))
        return sorted(num_set)

    def fetch(self, _list, _index):
        return [_list[item] for item in _index]

    def make_str(self, en_lines, zh_lines, score):
        en_lines = [item.strip() for item in en_lines]
        zh_lines = [item.strip() for item in zh_lines]
        en_txt = " ".join(en_lines)
        zh_txt = " ".join(zh_lines)

        item_txt = en_txt + "\n" + zh_txt + "\n" + str(score)

        return item_txt

    def write_corpus_to_file(self, corpus, file_name):
        print("writting to file...")
        with open(file_name, "a+", encoding="utf-8") as fd:
            fd.write("\n\n".join(corpus))
            fd.write("\n\n")

    def continue_pos_samples_func(self):
        for line_num in range(self.min_line_num, self.max_line_num):
            max_line = int(len(self.en_lines)/line_num) * line_num
            for start_line in range(0, max_line, line_num):
                end_line = start_line + line_num
                _range = list(range(start_line, end_line))
                en_lines = self.fetch(self.en_lines, _range)
                zh_lines = self.fetch(self.zh_lines, _range)
                yield self.make_str(en_lines, zh_lines, 1.0)

    def continue_pos_samples(self, num=100, file_name="pos.txt"):
        self.father_func(self.continue_pos_samples_func, num=num, file_name="corpus/avaliable/"+file_name)

    def discrete_pos_samples_func(self):
        for line_num in range(self.min_line_num, self.max_line_num):
            max_line = int(len(self.en_lines)/line_num) * line_num
            for start_line in range(0, max_line, line_num):
                rand_line_list = self.random((0, max_line-1), line_num)
                en_lines = self.fetch(self.en_lines, rand_line_list)
                zh_lines = self.fetch(self.zh_lines, rand_line_list)
                yield self.make_str(en_lines, zh_lines, 1.0)

    def discrete_pos_samples(self, num=100, file_name="pos.txt"):
        self.father_func(self.discrete_pos_samples_func, num=num, file_name="corpus/avaliable/"+file_name)

    def continue_neg_samples_func(self):
        for line_num in range(self.min_line_num, self.max_line_num):
            max_line = int(len(self.en_lines)/line_num) * line_num
            for start_line in range(0, max_line, line_num):
                end_line = start_line + line_num
                en_lines = self.en_lines[start_line: end_line]
                for _ in range(self.neg_num):
                    line_num_list = self.random((0, max_line-1), line_num, set(range(start_line, end_line)))
                    zh_lines = self.fetch(self.zh_lines, line_num_list)
                    yield self.make_str(en_lines, zh_lines, 0.0)

    def continue_neg_samples(self, num=100, file_name="neg.txt"):
        self.father_func(self.continue_neg_samples_func, num=num, file_name="corpus/avaliable/"+file_name)

    def discrete_neg_samples_func(self):
        for line_num in range(self.min_line_num, self.max_line_num):
            max_line = int(len(self.en_lines)/line_num) * line_num
            for _ in range(0, max_line, line_num):
                en_line_list = self.random((0, max_line-1), line_num)
                zh_line_list = self.random((0, max_line-1), line_num, set(en_line_list))
                en_lines = self.fetch(self.en_lines, en_line_list)
                zh_lines = self.fetch(self.zh_lines, zh_line_list)
                yield self.make_str(en_lines, zh_lines, 0.0)

    def discrete_neg_samples(self, num=100, file_name="neg.txt"):
        self.father_func(self.discrete_neg_samples_func, num=num, file_name="corpus/avaliable/"+file_name)

    def shuffle(self, _list):
        shuffle(_list)
        return _list

    def continue_mix_samples_func(self):
        for line_num in range(self.min_line_num, self.max_line_num):
            max_line = int(len(self.en_lines)/line_num) * line_num
            for start_line in range(0, max_line, line_num):
                end_line = start_line + line_num
                en_lines = self.en_lines[start_line: end_line]
                same_line = self.shuffle(list(range(start_line, end_line)))[:randint(0, line_num)]
                diff_line = self.random((0, max_line-1), line_num-len(same_line), block_list=set(range(start_line, end_line)))
                
                en = self.fetch(self.en_lines, range(start_line, end_line))
                zh = self.fetch(self.zh_lines, same_line) + self.fetch(self.zh_lines, diff_line)

                yield self.make_str(en, zh, len(same_line) * 1.0 / line_num)

    def continue_mix_samples(self, num=100, file_name="mix.txt"):
        self.father_func(self.continue_mix_samples_func, num=num, file_name="corpus/avaliable/"+file_name)

    def father_func(self, func, args=None, num=100, file_name="output.txt"):
        corpus = []
        corpus_size = 0
        item_num = 0
        for item_str in (func() if args is None else func(args)):
            corpus.append(item_str)
            corpus_size += len(item_str)
            item_num += 1
            if item_num >= num:
                break
            if corpus_size >= self.max_corpus_size:
                self.write_corpus_to_file(corpus, file_name)
                corpus = []
                corpus_size = 0
        if len(corpus) != 0:
            self.write_corpus_to_file(corpus, file_name)

    def gen_train_dataset(self, gen_num):
        print("generating training dataset...")
        self.cut(self.en_train_not_cut, self.zh_train_not_cut)
        self.load_file(self.en_train, self.zh_train)
        self.continue_pos_samples(gen_num)
        self.discrete_pos_samples(gen_num)
        self.continue_neg_samples(gen_num)
        self.discrete_neg_samples(gen_num)
        self.continue_mix_samples(gen_num)

    def gen_test_dataset(self, gen_num):
        print("generating test dataset...")
        self.cut(self.en_test_not_cut, self.zh_test_not_cut)
        self.load_file(self.en_test, self.zh_test)
        self.continue_pos_samples(gen_num, file_name="pos_test.txt")
        self.discrete_pos_samples(gen_num, file_name="pos_test.txt")
        self.continue_neg_samples(gen_num, file_name="neg_test.txt")
        self.discrete_neg_samples(gen_num, file_name="neg_test.txt")
        self.continue_mix_samples(gen_num, file_name="mix_test.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='corpus generation.')
    parser.add_argument('--mode', type=str, default="train", help="generating mode")
    args = parser.parse_args()

    generator = Generator()
    if args.mode == "train":
        generator.gen_train_dataset(2 * 10000)
    elif args.mode == "test":
        generator.gen_test_dataset(3000)
