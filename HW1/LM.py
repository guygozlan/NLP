from collections import Counter, defaultdict, OrderedDict
import numpy as np
import pandas as pd
import random

###############################

# Defint the base address of where the data is
BASE_DATA_FILES_ADDRESS = '/Users/guygozlan/Documents/Private/Binthomi/NLP/git/NLP/HW1/data/'
# Define the start and and symbols
START_SYMBOL = '<s>'
END_SYMBOL = '</s>'
# Define the file names for each language
LANGUAGE_LIST = ['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl']
# Train Vs. Test separation
DATA_SPLIT = {'train': 0.09, 'test': 0.01 }
# For Laplace smoothing - define the vocabulary size
VOCAB_SIZE = 2000
# Define weights
WEIGHTS_LIST = [0.4,0.3,0.3]

###############################

# Module supports going over given input data, counts number of occurrences for each n-gram
# And output module file according to defined format
# start/end symbols will be treated as single char
class LM:
    def __init__(self, data):
        self.input_data    = data.replace('\n','')
        self.start_symbol  = None
        self.end_symbol    = None
        self.start_n_gram  = None
        self.stop_n_gram   = None

    # Returns first char in str. Will treat start/end symbols as single char
    def get_next_char(self,str):
        if (str.find(self.start_symbol)==0):
            return (self.start_symbol)
        elif (str.find(self.end_symbol)==0):
            return (self.end_symbol)
        elif (len(str)==0):
            return (' ')
        else:
            return (str[0])

    # For given n-gram the function will count num of occurrences and return counter
    def count_occ(self, n_gram):
        counter = defaultdict(Counter)
        line = self.input_data
        while (len(line)):
            t=''
            for j in range(n_gram - 1):
                next_char = self.get_next_char(line)
                t += next_char
                line = line[len(next_char):]
            n_char = self.get_next_char(line)
            if (n_gram == 1): line = line[len(n_char):]
            counter[t][n_char] += 1
        return (counter)

    # The function will create counters list according to n-grams defined
    # And output the results to output file according to format definition
    # For unigram: We will smooth the data using Laplace
    def output_counters(self):
        self.counters_arr    = []
        self.n_gram_eval_arr = []
        str = ''

        for n_gram in range(self.start_n_gram, self.stop_n_gram):
            self.counters_arr.append(self.count_occ(n_gram))

        for i in reversed(range(self.stop_n_gram-self.start_n_gram)):
            str += '{}-gram:\n'.format(i+self.start_n_gram)
            keys = list(self.counters_arr[i].keys())
            for key in keys:
                occ_dict = self.counters_arr[i][key]
                sum_of_occurences = float(sum(occ_dict.values()))
                for chars, val in occ_dict.items():
                    concatenate_chars = key+chars
                    prob = val/sum_of_occurences
                    # Add One (Laplace) for unigram
                    if (i==1): prob = (val+1)/(sum_of_occurences+VOCAB_SIZE)
                    log_prob = np.log2(prob)
                    str+='{}\t{}\n'.format(concatenate_chars, log_prob)
            str+='\n'
        return(str)

class Eval(object):
    def __init__(self, input_data):
        self.start_symbol  = None
        self.end_symbol    = None
        self.weights       = None
        self.input_data    = input_data

    # This funtion counts the num of n-grams in model file. For out task - should be 3
    @staticmethod
    def count_n_grams(model_file):
        with open(model_file,'r') as file:
            count = 0
            for line in file:
                if (line[line.find('-'):]=='-gram:\n'): count+=1
        return (count)

    # Returns first char in str. Will treat start/end symbols as single char
    def get_next_char(self, str):
        if (0 == str.find(self.start_symbol)): return self.start_symbol
        if (0 == str.find(self.end_symbol))  : return self.end_symbol
        return (str[0])

    # Function creates 3 OrderedDict, 1 for each n-gram.
    # Dict contains tuple:Probability_Val
    def model_file_to_dict(self, model_file):
        self.n_gram_dict = OrderedDict()
        self.n_gram_dict['3-gram'] = OrderedDict()
        self.n_gram_dict['2-gram'] = OrderedDict()
        self.n_gram_dict['1-gram'] = OrderedDict()

        with open(model_file, 'r') as f:
            line = f.readline()
            if (line != '3-gram:\n'): raise ValueError('Model file is not in correct format! Excpected 3-gram')
            curr_n_gram = 3
            for line in f:
                if ('\n' == line): continue
                if ('-gram:\n' in line):
                    curr_n_gram = int(line.split('-')[0])
                    continue
                chars, logPr = line.replace('\n', '').split('\t')
                self.n_gram_dict['{}-gram'.format(curr_n_gram)][chars] = 2**float(logPr)

    # Calculates the P_interpolation according to given Lambda array for each 3-chars in model file
    def calc_Pinterpolation(self, model_file):
        self.model_file_to_dict(model_file)
        self.interpolation_dict = OrderedDict()
        for trigram_chars, trigram_Pr in self.n_gram_dict['3-gram'].items():
            # Calculate the substrings to search in unigram and bigram models
            bigram_str  = trigram_chars[len(self.get_next_char(trigram_chars)):]
            unigram_str = bigram_str[len(self.get_next_char(bigram_str)):]
            # Search the probability in the unigram and biagram models
            bigram_Pr  = self.n_gram_dict['2-gram'][bigram_str]
            unigram_Pr = self.n_gram_dict['1-gram'][unigram_str]
            # Calculate the interpolation and store in self.interpolation_dict
            Pinterp = self.weights[0] * trigram_Pr + self.weights[1] * bigram_Pr + self.weights[2] * unigram_Pr
            self.interpolation_dict[''.join(trigram_chars)] = Pinterp

    # Going over the input char by char - creates the P_interp according to formula
    # For cases where we have unfamiliar data - use Laplace for smoothing
    # Then, calculate the preplexity according to formula
    def calc_preplexity(self):
        line = self.input_data
        sum_log_P_interp = 0.0
        while (len(line)>(1+len(self.end_symbol))):
            t = ''
            for j in range(3):
                next_char = self.get_next_char(line)
                t += next_char
                line = line[len(next_char):]
            try:
                sum_log_P_interp += np.log2(self.interpolation_dict[t])
            except:
                # Add One (Laplace) for unigram: If string was not found: Pinterp=(1/VOCAB_SIZE)*unigram_weight + 0 + 0
                sum_log_P_interp += np.log2(self.weights[2] * 1/VOCAB_SIZE)
            # Skip empty end
            if (line == self.start_symbol+self.end_symbol): break

        # Calculate the preplexity
        H = -1 * sum_log_P_interp/(len(self.input_data)-2)
        Preplexity = 2**H
        return(Preplexity)

############################

# The function supports removing links (https://) and tags (@)
# This will increase preplexity
def clean_row_data(str):
    ret = str
    # Clear links
    while ('https://' in ret):
        ret = ret[:ret.find('https://')] + ' '.join(ret[ret.find('https://'):].split()[1:])
    # Clear tags
    while ('@' in ret):
        ret = ret[:ret.find('@')] + ' '.join(ret[ret.find('@'):].split()[1:])

    return(ret)

# Assuming input is CSV file where column 'tweet_test' is the data
# Combining all given data to one long string with seperations of start/end symbols over each line
def preproc_csv_file(csv_file, start_symbol, end_symbol, clean_data = False):
    df = pd.read_csv(csv_file)
    data = start_symbol
    for line in df['tweet_text']:
        if (len(line)==0): continue
        if (clean_data):
            line = clean_row_data(line)
        data+='{}{}{}'.format(line, end_symbol, start_symbol)
    data+=end_symbol
    return(data)

# From each language in LANGUAGE_LIST
# Split the data according to DATA_SPLIT
# Creates training and test CSVs as and output
def split_to_train_and_test_csvs():
    for lang in LANGUAGE_LIST:
        file_size = sum(1 for line in open(BASE_DATA_FILES_ADDRESS + lang + '.csv'))
        for data_split in ['train', 'test']:
            desired_data_size = int(file_size * DATA_SPLIT[data_split])
            skip = sorted(random.sample(xrange(1,file_size+1), file_size-desired_data_size))
            df = pd.read_csv(BASE_DATA_FILES_ADDRESS + lang + '.csv', skiprows=skip)
            df.to_csv(BASE_DATA_FILES_ADDRESS + lang + '_' + data_split + '.csv')

###############################

# Using corpus file - generate LM model file according to format
def lm(corpus_file, model_file):
    input_data = preproc_csv_file(corpus_file, START_SYMBOL, END_SYMBOL, clean_data=True)

    lm_m = LM(input_data)
    lm_m.start_symbol = START_SYMBOL
    lm_m.end_symbol   = END_SYMBOL
    lm_m.start_n_gram = 1
    lm_m.stop_n_gram  = 4

    output = lm_m.output_counters()

    with open(model_file, 'w+') as f_output:
        f_output.write(output)

# Calculate preplexity with model and weights and applying on the input file
# Return the preplexity
def eval(input_file, model_file, weights):
    input_data = preproc_csv_file(input_file, START_SYMBOL, END_SYMBOL)
    eval_m = Eval(input_data)
    eval_m.start_symbol = START_SYMBOL
    eval_m.end_symbol   = END_SYMBOL
    eval_m.weights      = weights
    eval_m.calc_Pinterpolation(model_file)
    preplexity = eval_m.calc_preplexity()
    return(preplexity)

# Pre-proccess the data into seperated CSVs
# For each cell in DF: calculate preplexity
# return DataFram results of all permutations
def run():
    # Split the data into test and training sets CSVs
    split_to_train_and_test_csvs()
    results = pd.DataFrame(columns=LANGUAGE_LIST, index=LANGUAGE_LIST)
    for lang_train in LANGUAGE_LIST:
        for lang_test in LANGUAGE_LIST:
            train_input_file_path = BASE_DATA_FILES_ADDRESS + lang_train + '_train.csv'
            test_input_file_path  = BASE_DATA_FILES_ADDRESS + lang_test  + '_test.csv'
            output_file_path      = BASE_DATA_FILES_ADDRESS + lang_test  + '_model.txt'
            lm(train_input_file_path, output_file_path)
            preplexity = eval(test_input_file_path, output_file_path, WEIGHTS_LIST)
            results[lang_test][lang_train] = preplexity
    print (results)


###############################

run()

