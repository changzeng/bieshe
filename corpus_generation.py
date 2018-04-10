# encoding: utf-8

from random import randint
from sys import getsizeof


class Mixer(object):
    def __init__(self):
        self.en = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.en"
        self.zh = "corpus/raw/chinese_and_english/news-commentary-v12.zh-en.zh"
        self.load_file()
        self.mix()

    def load_file(self):
        files = [self.en, self.zh]
        for _file in files:
            with open(_file, encoding="utf-8", errors="ignore") as fd:
                lines = fd.read().strip().split("\n")
                lines = [line.strip() for line in lines]
                if "zh" == _file[-2:]:
                    self.zh_lines = lines
                elif "en" == _file[-2:]:
                    self.en_lines = lines

    def mix(self):
        def make_str(en_lines, zh_lines):
            en_txt = "".join(en_lines)
            zh_txt = "".join(zh_lines)

            item_txt = en_txt + "\n" + zh_txt

            return item_txt

        corpus_list = []
        corpus_size = 0
        line_num = len(self.en_lines)

        # print(len(self.en_lines))
        # print(len(self.zh_lines))
        # input()

        # check_line = int((line_num-1)/2/2/2/2)
        # check_line += int(check_line/8*4)
        # check_line = 22015
        # print(self.en_lines[check_line])
        # print(self.zh_lines[check_line])
        # input()

        # start_line = check_line
        # for check_line in range(start_line, line_num):
        #     print(check_line)
        #     print(self.en_lines[check_line])
        #     print(self.zh_lines[check_line])
        #     input()

        max_line = 2000
        min_line = 10
        for file_size in range(min_line, max_line, 2):
            print("process progress... (file size)/(max_line) %5d/%5d, corpus list size %6dMB" % (file_size, max_line, corpus_size/(1024*1024)))
            for start_line in range(0, line_num, file_size):
                # 抽取连续的中英字符串
                en_lines = self.en_lines[start_line: start_line+file_size]
                zh_lines = self.zh_lines[start_line: start_line+file_size]

                corpus_size += getsizeof(en_lines) + getsizeof(zh_lines)
                
                item_txt = make_str(en_lines, zh_lines)
                corpus_list.append(item_txt)
                
                # 随机抽取中英字符串
                en_lines = []
                zh_lines = []
                for i in range(file_size):
                    index = randint(0, line_num-1)

                    en_line = self.en_lines[index]
                    zh_line = self.zh_lines[index]

                    en_lines.append(en_line)
                    zh_lines.append(zh_line)

                    corpus_size += len(en_line) + len(zh_line)
                
                item_txt = make_str(en_lines, zh_lines)
                corpus_list.append(item_txt)

                if corpus_size >= 50 * 1024 * 1024:
                    print("writting to file...")
                    with open("output.txt", "a", encoding="utf-8") as fd:
                        fd.write("\n\n".join(corpus_list))
                        fd.write("\n\n")
                    corpus_list = []
                    corpus_size = 0


if __name__ == "__main__":
    mixer = Mixer()
