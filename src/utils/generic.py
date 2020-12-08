import sys

import numpy as np 

def prints(content, status):
    if status is 0:
        msg = '.'
    elif status is 1:
        msg = 'DONE'
    elif status is 2:
        msg = 'WARN'
    elif status is 3:
        msg = 'FAIL'
    print(f'\n[{msg}]\t{content}\n')

class ProgressNotifier():

    def __init__(self, total:int, title='', bar_len:int=20, show_bar=True):
        self.total = total
        self.title = title + ' '
        self.bar_len = bar_len
        self.show_bar = show_bar
        self.current = 1

    def reset(self):
        self.current = 1

    def update(self):
        fraction = float(self.current) / self.total
        n_done = int(fraction*self.bar_len)

        bar = self.title
        numdigits = int(np.log10(self.total)) + 1
        bar += ('%' + str(numdigits) + 'd/%d') % (self.current, self.total)

        if self.show_bar:
            bar += ' ['
            if n_done > 0:
                bar += ('=' * (n_done - 1))
                if self.current < self.total:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.bar_len - n_done))
            bar += ']'

        bar += (' {}%'.format(int(fraction*100)))

        self.current += 1
        
        sys.stdout.write('\b' * self.total)
        sys.stdout.write('\r')
        sys.stdout.write(bar)
        sys.stdout.flush()