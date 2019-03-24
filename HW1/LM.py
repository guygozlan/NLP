from collections import Counter, defaultdict
import numpy as np

CHARS_TO_REMOVE = "\r\n"

class LM:
    def __init__(self, data):
        self.input_data    = data.translate(None, CHARS_TO_REMOVE)
        self.start_symbol  = None
        self.end_symbol    = None
        self.start_n_gram  = None
        self.stop_n_gram   = None

    def get_next_char(self,str):
        if (str.find(self.start_symbol)==0):
            return (self.start_symbol)
        elif (str.find(self.end_symbol)==0):
            return (self.end_symbol)
        elif (len(str)==0):
            return (' ')
        else:
            return (str[0])

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
                    log_prob = np.log(prob)
                    str+='{}\t{}\n'.format(concatenate_chars, log_prob)
            str+='\n'
        return(str)

class M3Gram:
    w_i    = None
    w_im1  = None
    w_im2  = None
    log2Px = None
    Px     = None
class M2Gram:
    w_i    = None
    w_im1  = None
    log2Px = None
    Px     = None
class M2Gram:
    w_i    = None
    log2Px = None
    Px     = None

class Eval(object):
    def __init__(self):
        self.start_symbol  = None
        self.end_symbol    = None

    @staticmethod
    def count_n_grams(model_file):
        with open(model_file,'r') as file:
            count = 0
            for line in file:
                if (line[line.find('-'):]=='-gram:\n'): count+=1
        return (count)

    def get_next_char(self, str):
        if (0 == str.find(self.start_symbol)): return self.start_symbol
        if (0 == str.find(self.end_symbol))  : return self.end_symbol
        return (str[0])

    def create_3_gram_matrix(self,model_file):
        self.m_3_gram = []
        with open(model_file, 'r') as file:
            if (file.readline() != '3-gram:\n'): raise ValueError('model_file is not is format. Could not find 3-gram')
            for line in file:
                if (line == '\n'): break
                str, log2Px = line.split('\t')
                m_3_gram_el = M3Gram()
                m_3_gram_el.w_im2  = self.get_next_char(str)
                m_3_gram_el.w_im1  = self.get_next_char(str[len(m_3_gram_el.w_im2):])
                m_3_gram_el.w_i    = self.get_next_char(str[len(m_3_gram_el.w_im2)+len(m_3_gram_el.w_im1):])
                m_3_gram_el.log2Px = float(log2Px)
                m_3_gram_el.Px     = 2**m_3_gram_el.log2Px
                self.m_3_gram.append(m_3_gram_el)

    def create_2_gram_matrix(self, model_file):
        self.m_2_gram = []
        with open(model_file, 'r') as file:
            line = file.readline()
            while (line != '2-gram:\n'): line = file.readline()
            for line in file:
                if (line == '\n'): break
                str, log2Px = line.split('\t')
                m_2_gram_el = M2Gram()
                m_2_gram_el.w_im1  = self.get_next_char(str)
                m_2_gram_el.w_i    = self.get_next_char(str[len(m_2_gram_el.w_im1):])
                m_2_gram_el.log2Px = float(log2Px)
                m_2_gram_el.Px     = 2 ** m_2_gram_el.log2Px
                self.m_2_gram.append(m_2_gram_el)

    def create_1_gram_matrix(self, model_file):
        self.m_1_gram = []
        with open(model_file, 'r') as file:
            line = file.readline()
            while (line != '1-gram:\n'): line = file.readline()
            for line in file:
                if (line == '\n'): break
                str, log2Px = line.split('\t')
                m_1_gram_el        = M2Gram()
                m_1_gram_el.w_i    = self.get_next_char(str)
                m_1_gram_el.log2Px = float(log2Px)
                m_1_gram_el.Px     = 2 ** m_1_gram_el.log2Px
                self.m_1_gram.append(m_1_gram_el)

    def get2GramElement(self, w_i, w_im1):
        for el in self.m_2_gram:
            if (el.w_i==w_i and el.w_im1==w_im1): return el
        return None

    def get1GramElement(self, w_i):
        for el in self.m_2_gram:
            if (el.w_i==w_i): return el
        return None


###############################

def lm(corpus_file, model_file):
    with open(corpus_file, 'r') as f_input:
      input_data = f_input.read()

    lm_m = LM(input_data)
    lm_m.start_symbol = '<start>'
    lm_m.end_symbol   = '<end>'
    lm_m.start_n_gram = 1
    lm_m.stop_n_gram  = 4

    output = lm_m.output_counters()
    #print (output)

    with open(model_file, 'w+') as f_output:
        f_output.write(output)

def eval(input_file, model_file, weights):
    eval_m = Eval()
    eval_m.start_symbol = '<start>'
    eval_m.end_symbol   = '<end>'

    number_of_n_grams = eval_m.count_n_grams(model_file)
    if (number_of_n_grams != len(weights)): raise ValueError('Number of weights does not match number of n_grams in the model file')
    if (float(sum(weights)) != 1.0):        raise ValueError('Weights need to sum to exactly 1.0')

    lambda_3_gram = weights[2]
    lambda_2_gram = weights[1]
    lambda_1_gram = weights[0]

    eval_m.create_3_gram_matrix(model_file)
    eval_m.create_2_gram_matrix(model_file)
    eval_m.create_1_gram_matrix(model_file)

    Accum_entropy = 0.0
    for el_3_gram in eval_m.m_3_gram:
        P_interp = lambda_3_gram * el_3_gram.Px + lambda_2_gram * eval_m.get2GramElement(el_3_gram.w_i, el_3_gram.w_im1).Px + lambda_1_gram * eval_m.get1GramElement(el_3_gram.w_i).Px
        Accum_entropy -= P_interp*np.log(P_interp)

    print ('Entropy = {}'.format(Accum_entropy))
    print ('Perplexity = {}'.format(2**Accum_entropy))

###############################

corpus_file = '/Users/guygozlan/Documents/Private/Binthomi/NLP/git/NLP/HW1/input_file_1.txt'
model_file  = '/Users/guygozlan/Documents/Private/Binthomi/NLP/git/NLP/HW1/output_file_1.txt'

lm(corpus_file, model_file)
eval(corpus_file, model_file, [0.2,0.3,0.5])
