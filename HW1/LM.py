from collections import Counter, defaultdict
import numpy as np

start_symbol = '<start>'
end_symbol = '<end>'
start_n_gram=1
stop_n_gram=4
charsToRemove = "\r\n"

class LM:
    def __init__(self, data):
        self.input_data = data.translate(None, charsToRemove)

    @staticmethod
    def get_next_char(str):
        if (str.find(start_symbol)==0):
            return (start_symbol)
        elif (str.find(end_symbol)==0):
            return (end_symbol)
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

    def output_counters(self, start_n_gram, stop_n_gram):
        self.counters_arr    = []
        self.n_gram_eval_arr = []
        str = ''

        for n_gram in range(start_n_gram, stop_n_gram):
            self.counters_arr.append(self.count_occ(n_gram))

        for i in reversed(range(stop_n_gram-start_n_gram)):
            str += '{}-gram:\n'.format(i+start_n_gram)
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

    output = lm_m.output_counters(start_n_gram=start_n_gram, stop_n_gram=stop_n_gram)
    print (output)

    with open(model_file, 'w+') as f_output:
        f_output.write(output)

###############################

corpus_file = '/Users/guygozlan/Documents/Private/Binthomi/NLP/git/NLP/HW1/input_file_1.txt'
model_file  = '/Users/guygozlan/Documents/Private/Binthomi/NLP/git/NLP/HW1/output_file_1.txt'

lm(corpus_file, model_file)
