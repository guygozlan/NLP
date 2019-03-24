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
                    str+='{:20}{}\n'.format(concatenate_chars, log_prob)
            str+='\n'
        return(str)

def lm(corpus_file, model_file):
    with open(corpus_file, 'r') as f_input:
      input_data = f_input.read()

    lm_m = LM(input_data)
    lm_m.start_symbol = '<start>'
    lm_m.end_symbol = '<end>'
    lm_m.start_n_gram = 1
    lm_m.stop_n_gram = 4

    output = lm_m.output_counters()
    print (output)

    with open(model_file, 'w+') as f_output:
        f_output.write(output)

###############################

corpus_file = '/Users/guygozlan/Documents/Private/Binthomi/NLP/git/NLP/HW1/input_file_1.txt'
model_file  = '/Users/guygozlan/Documents/Private/Binthomi/NLP/git/NLP/HW1/output_file_1.txt'

lm(corpus_file, model_file)
