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

    def update(self, metrics:dict={}):

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

        # Add fraction as percentage
        bar += (' {}%'.format(int(fraction*100)))

        # Add metrics
        if metrics != {}:
            bar += ' |'
            for key in metrics.keys():
                bar += ' {}: {:.2f}'.format(key, float(metrics[key]))

        self.current += 1
        
        sys.stdout.write('\b' * 200)#self.total)
        sys.stdout.write('\r')
        sys.stdout.write(bar)
        sys.stdout.flush()

class DataStream():
    def __init__(self, data, labels, name='Unnamed DataStream'):

        assert data.shape[0] == labels.shape[0], \
            "Data and labels do not fit in shape. Got data with shape {} and"\
            "labels with shape {}. The first dimension has to be the same"\
            .format(data.shape, labels.shape)

        self.data = data
        self.labels = labels
        self.name = name
        
        self.size = data.shape[0]
        self.index = 0

    def next(self):
        """ Get the datapoint at the current index and increase the index. 
        If the index has reached the end of the dataset, raise an IndexError 
        and notify that index has to be reset. """

        # Fail safe if index has reached end of dataset
        try:
            # Draw the current image
            item = self.data[self.index]
        except IndexError:
            raise IndexError(
                "{} reached the end of its data. To use further, "
                "call `reset()` to reset the index to 0.".format(name))

        # Increase the index
        self.index += 1

        return item

    def reset(self):
        self.index = 0

