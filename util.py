# encoding: utf-8
# author: liaochangzeng
# e-mail: 1207338829@qq.com


# 判断一个字符是不是数字
def is_number(char):
    if 48 <= ord(char) <= 57:
        return True
    return False