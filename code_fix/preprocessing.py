import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import glob
import collections
import glob
import json
import os
import random
import re
import sys
from numpy.random import rand


Token = collections.namedtuple('Token', ['type', 'value', 'line', 'column'])
keywords = ['auto', 'break', 'case', 'const', 'continue', 'default',
            'do', 'else', 'enum', 'extern', 'for', 'goto', 'if',
            'register', 'return', 'signed', 'sizeof', 'static', 'switch',
            'typedef', 'void', 'volatile', 'while', 'EOF', 'NULL',
            'null', 'struct', 'union']
includes = ['stdio.h', 'stdlib.h', 'string.h', 'math.h', 'malloc.h',
            'stdbool.h', 'cstdio', 'cstdio.h', 'iostream', 'conio.h']
calls = ['printf', 'scanf', 'cin', 'cout', 'clrscr', 'getch', 'strlen',
         'gets', 'fgets', 'getchar', 'main', 'malloc', 'calloc', 'free']
types = ['char', 'double', 'float', 'int', 'long', 'short', 'unsigned']


class NMTDataset:
    def __init__(self):
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None


    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    ## Step 1 and Step 2
    def preprocess_code(self, path):
        # w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        # w = re.sub(r"([?.!,¿])", r" \1 ", w)
        # w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        # w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.

        result = self.code_to_tok(path)
        lines = []
        res = ''
        for line in result:
            for c in line:
                res += c
                if c != '<eof>':
                    res += ' '
            lines.append(res)
            res = ''
        return lines

    def escape(self, string):
        return repr(string)[1:-1]

    def tokenize_code(self, code):
        keywords = {'IF', 'THEN', 'ENDIF', 'FOR', 'NEXT', 'GOSUB', 'RETURN'}
        token_specification = [
            ('comment',
             r'\/\*(?:[^*]|\*(?!\/))*\*\/|\/\*([^*]|\*(?!\/))*\*?|\/\/[^\n]*'),
            ('directive', r'#\w+'),
            ('string', r'"(?:[^"\n]|\\")*"?'),
            ('char', r"'(?:\\?[^'\n]|\\')'"),
            ('char_continue', r"'[^']*"),
            ('number', r'[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
            ('include', r'(?<=\#include) *<([_A-Za-z]\w*(?:\.h))?>'),
            ('op',
             r'\(|\)|\[|\]|{|}|->|<<|>>|\*\*|\|\||&&|--|\+\+|[-+*|&%\/=]=|[-<>~!%^&*\/+=?|.,:;#]'),
            ('name', r'[_A-Za-z]\w*'),
            ('whitespace', r'\s+'),
            ('nl', r'\\\n?'),
            ('MISMATCH', r'.'),  # Any other character
        ]
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)

        line_num = 1
        line_start = 0
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'NEWLINE':
                line_start = mo.end()
                line_num += 1
            elif kind == 'SKIP':
                pass
            elif kind == 'MISMATCH':
                return
            else:
                if kind == 'ID' and value in keywords:
                    kind = value
                column = mo.start() - line_start
                yield Token(kind, value, line_num, column)

    def tokenize_code_to_tok(self, code, keep_names=True, ut=False, keep_lines=False):
        names = ''
        line_count = 1
        name_dict = {}
        name_sequence = []

        regex = '%(d|i|f|c|s|u|g|G|e|p|llu|ll|ld|l|o|x|X)'
        isNewLine = True

        my_gen = self.tokenize_code(code)

        tokenized_code_split = ['<sof>']
        result = '<sof>'

        if keep_lines:
            tokenized_code_split.append('0 ~ ')
            result += '0 ~ '
        else:
            tokenized_code_split.append('~')
            result += '~ '

        while True:
            try:
                token = next(my_gen)
            except StopIteration:
                break

            if isinstance(token, Exception):
                return '', '', ''

            type_ = str(token[0])
            value = str(token[1])

            if value in keywords:
                result += '_<keyword>_' + self.escape(value) + ' '
                tokenized_code_split.append(self.escape(value))
                isNewLine = False

            elif type_ == 'include':
                result += '_<include>_' + self.escape(value).lstrip() + ' '
                tokenized_code_split.append(self.escape(value).lstrip())
                isNewLine = False

            elif value in calls:
                result += '_<APIcall>_' + self.escape(value) + ' '
                tokenized_code_split.append(self.escape(value))
                isNewLine = False

            elif value in types:
                result += '_<type>_' + self.escape(value) + ' '
                tokenized_code_split.append(self.escape(value))
                isNewLine = False

            elif type_ == 'whitespace' and (('\n' in value) or ('\r' in value)):
                if isNewLine:
                    continue

                if keep_lines:
                    result += ' '.join(list(str(line_count))) + ' ~ '
                    tokenized_code_split.append(str(line_count) + ' ~')
                else:
                    result += ' ~ '
                    tokenized_code_split.append('~')
                line_count += 1
                isNewLine = True

            elif type_ == 'whitespace' or type_ == 'comment' or type_ == 'nl':
                pass

            elif 'string' in type_:
                matchObj = [m.group().strip()
                            for m in re.finditer(regex, value)]
                result += '_<string>_' + ' '
                tokenized_code_split.append('_<string>_')
                isNewLine = False

            elif type_ == 'name':
                if keep_names:
                    if self.escape(value) not in name_dict:
                        name_dict[self.escape(value)] = str(
                            len(name_dict) + 1)

                    name_sequence.append(self.escape(value))
                    result += '_<id>_' + name_dict[self.escape(value)] + '@ '
                    names += '_<id>_' + name_dict[self.escape(value)] + '@ '
                    tokenized_code_split.append(name_dict[self.escape(value)] + '@')
                else:
                    result += '_<id>_' + '@ '
                    tokenized_code_split.append('_<id>_' + '@')
                isNewLine = False

            elif type_ == 'number':
                result += '_<number>_' + '# '
                tokenized_code_split.append('_<number>_' + '# ')
                isNewLine = False

            elif 'char' in type_ or value == '':
                result += '_<' + type_.lower() + '>_' + ' '
                tokenized_code_split.append(type_.lower())
                isNewLine = False

            else:
                converted_value = self.escape(value).replace('~', 'TiLddE')
                result += '_<' + type_ + '>_' + converted_value + ' '
                tokenized_code_split.append(converted_value)

                isNewLine = False

        result = result[:-1]
        names = names[:-1]

        if result.endswith('~'):
            idx = result.rfind('}')
            result = result[:idx + 1]

        tokenized_code_split.append('<eof>')

        return tokenized_code_split  # sanitize_brackets(result) name_dict, name_sequence,

    def code_to_tok(self, path):
        codes = []
        a = 0
        for filename in glob.glob(path):
            if a % 1000 == 0:
                print(filename)
            a += 1
            if a > 25000:
                break
            with open(filename, 'r') as f:
                raw_code = f.read()
                # tokenized_code_split = self.tokenize_code_to_tok(raw_code)
                # codes.append(tokenized_code_split)
                for line in raw_code.splitlines():
                    tokenized_code_split = self.tokenize_code_to_tok(line)
                    codes.append(tokenized_code_split)
        return codes

    def make_typos(self, tokens):
        if rand() < 0.2:
            return tokens
        typo_tokens = ['(', ')', '[', ']', ';', ',']
        remove_tok = []
        for tok in typo_tokens:
            if tok not in tokens:
                remove_tok.append(tok)
        for tok in remove_tok:
            typo_tokens.remove(tok)
        if len(typo_tokens) == 0:
            return tokens
        rand_token_idx = random.randint(0, len(typo_tokens) - 1)
        rand_action = 0
        if typo_tokens[rand_token_idx] != ';':
            rand_action = random.randint(0, 1)  # 0 - delete, 1 - duplicate
        token = typo_tokens[rand_token_idx]

        num_of_tokens = 0
        for tok in tokens:
            if token == tok:
                num_of_tokens += 1
        rand_replace = 1
        if num_of_tokens > 1:
            rand_replace = random.randint(1, num_of_tokens)

        idx = 0
        count = 0
        for i in range(len(tokens)):
            if tokens[i] == token:
                count += 1
                idx = i
            if count == rand_replace:
                break

        if rand_action == 0:
            tokens.pop(idx)
        elif rand_action == 1:
            tokens.insert(idx, token)

        return tokens

    def split_to_str(self, split):
        res = ''
        for tok in split:
            if res != '':
                res += ' '
            res += tok
        return res

    def create_dataset(self, path):
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        #lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        #word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

        #return zip(*word_pairs)

        correct_code = self.code_to_tok(path)

        not_correct_code = []
        # a = 0
        for code in correct_code:
            copy = code[:]
            not_correct_code.append(self.make_typos(copy))
            # print(len(correct_code[a]), " = ", len(not_correct_code[a]))
            # a += 1

        correct_code_str = []
        for code in correct_code:
            correct_code_str.append(self.split_to_str(code))

        not_correct_code_str = []
        for code in not_correct_code:
            not_correct_code_str.append(self.split_to_str(code))

        res = []
        for i in range(len(correct_code_str)):
            temp = []
            temp.append(correct_code_str[i])
            temp.append(not_correct_code_str[i])
            res.append(temp)

        # print(correct_code_str[0])
        # print(not_correct_code_str[0])
        # print()
        # print(correct_code_str[1])
        # print(not_correct_code_str[1])
        # print()
        # print(correct_code_str[2])
        # print(not_correct_code_str[2])
        # print()
        # print(correct_code_str[3])
        # print(not_correct_code_str[3])
        # print()
        # print(correct_code_str[4])
        # print(not_correct_code_str[4])
        # print()
        # print(correct_code_str[5])
        # print(not_correct_code_str[5])
        # print()
        # print(correct_code_str[6])
        # print(not_correct_code_str[6])
        # print()
        # print(correct_code_str[7])
        # print(not_correct_code_str[7])
        # print()
        # print(correct_code_str[8])
        # print(not_correct_code_str[8])
        #
        # aaaa = zip(*res)

        return zip(*res)

    # Step 3 and Step 4
    def tokenize(self, lang):
        # lang = list of sentences in a language

        # print(len(lang), "example sentence: {}".format(lang[0]))
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(lang)

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn)
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(lang)

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences
        ## and pads the sequences to match the longest sequences in the given input
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, path, num_examples=None):
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset(path)

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def call(self, num_examples, BUFFER_SIZE, BATCH_SIZE, file_path):
        #file_path = "small\\*.txt"
        input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(file_path, num_examples)

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer

    def create_dataset_predict(self, path):
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        #lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        #word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

        #return zip(*word_pairs)

        predict_code = self.code_to_tok(path)

        predict_code_str = []
        for code in predict_code:
            predict_code_str.append(self.split_to_str(code))

        return predict_code_str

    def load_dataset_predict(self, path):
        # creating cleaned input, output pairs
        input_code = self.create_dataset_predict(path)

        input_tensor, input_tokenizer = self.tokenize(input_code)

        return input_tensor, input_tokenizer

    def call_predict(self, file_path):
        input_tensor, self.input_tokenizer = self.load_dataset_predict(file_path)

        #input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        #train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        #train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        #val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        #val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        #return train_dataset, val_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer
        return input_tensor, self.input_tokenizer


unique_tok = []
with open("unique_tok.txt", 'r') as ff:
    for word in ff:
        word = re.sub('\n', '', word)
        unique_tok.append(word)

BUFFER_SIZE = 32000
BATCH_SIZE = 32
num_examples = 50000

dataset_creator = NMTDataset()
train_dataset, val_dataset, inp_tok, targ_tok = dataset_creator.call(num_examples, BUFFER_SIZE, BATCH_SIZE, "small\\*.txt")
print()