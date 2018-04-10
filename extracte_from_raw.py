# encoding : utf-8
from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from functools import reduce
from io import StringIO
from io import open
import re, langid

input_fold = "corpus/raw/"
output_fold = "corpus/avaliable/"


def txt_filter(txt):
    txt = txt.replace("学英语，找付帅|微信：fushuailaoshi", "")
    txt = txt.replace("\n", "")

    return txt


def is_chinese(txt):
    sub_txt = txt

    english_symbol = set(["’", ".", ","])
    chinese_symbol = set(["、"])

    length = len(sub_txt)
    alpha_count = 0
    for item in sub_txt:
        # print(item, item.isupper() or item.islower(), sep=" ")
        # input()
        if item in english_symbol:
            return False

        if item in chinese_symbol:
            return True

        if item.isupper() or item.islower() or item == " ":
            alpha_count += 1

    if alpha_count/length >= 0.5:
        return False

    return True


def remove_blank(txt):
    txt = txt.replace("\r\n", "")
    txt = txt.replace(" ", "")
    txt = txt.replace("\t", "")

    return txt


def FileWrapper(func):
    def wrapper(input_file, output_file):
        txt = readText(input_fold + input_file)

        result = func(txt)

        saveTxt(output_fold + output_file,
                reduce(lambda x, y: x + "\n\n" + y, map(lambda x: x[0].strip() + "\n" + x[1].strip(), result)))

    return wrapper


def readText(file_name):
    try:
        with open(file_name, "rb") as pdf_file:
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, laparams=laparams)

            process_pdf(rsrcmgr, device, pdf_file)
            device.close()

            content = retstr.getvalue()
            retstr.close()
            return content
    except:
        with open(file_name, "r", encoding="gbk") as fd:
            return fd.read()


def saveTxt(txt_file, content):
    with open(txt_file, "w") as f:
        f.write(content)


@FileWrapper
def six_level_translation(txt):
    txt = txt_file(txt)
    p = re.compile("\d+、")

    passage_list = p.split(txt)
    # print(passage_list[1])

    pattern = "。 [a-zA-Z]"
    p = re.compile(pattern)

    result = []

    for i, passage in enumerate(passage_list):
        passage = passage.strip()
        if len(passage) == 0:
            continue

        s = p.search(passage).span()

        chinese = remove_blank(passage[:s[0] + 1])
        english = passage[s[1] - 1:]

        result.append((english, chinese))

    return result


@FileWrapper
def kaoyan_reading(txt):
    txt = txt_filter(txt)
    # print(txt)
    result = []
    pattern = "\d{4} Text \d{1,2}"
    r = re.compile(pattern)

    passage_list = r.split(txt)

    pattern = "[\u4e00-\u9fa5]"
    r = re.compile(pattern)

    for passage in passage_list:
        passage = passage.strip()
        if len(passage) == 0:
            continue

        s = r.search(passage).span()

        english = passage[:s[0]]
        chinese = remove_blank(passage[s[1] - 1:])

        result.append((english, chinese))

    return result


@FileWrapper
def politics_transfer(txt):
    r = re.compile("\n+")
    txt = re.sub(r, "\n", txt)
    txt = txt.split("******")

    result = []

    for item in txt:
        if item == None:
            continue

        chinese = []
        english = []

        txt_list = item.split("\n")

        for i, txt_item in enumerate(txt_list):
            if len(txt_item[:3]) == 0:
                continue
            # print("hello %d" % i)
            if is_chinese(txt_item):
                chinese.append(txt_item)
            else:
                if type(txt_item) != str:
                    continue
                english.append(txt_item)
            # print("world %d" % i)

        english = " ".join(english)

        r = re.compile(" +")
        english = re.sub(r, " ", english)

        chinese = "".join(chinese).replace(" ", "")

        result.append((english, chinese))

    return result


if __name__ == "__main__":
    if False:
        files = ["six_translation.pdf"]

        for _file in files:
            six_level_translation(_file, _file.split(".")[0])
    if False:
        files = ["kaoyan_reading.pdf"]

        for _file in files:
            kaoyan_reading(_file, _file.split(".")[0])
    if True:
        files = ["政治文献20121128.txt", "政治文献20121229.txt"]

        for _file in files:
            politics_transfer(_file, _file.split(".")[0])
