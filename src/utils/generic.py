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

    def __init__(self, total:int, title=''):
        self.total = total
        self.title = title + ' '
        self.index = 1

    def reset(self):
        self.index = 1

    def update(self):

        fraction = float(self.index) / self.total
        output = self.title
        numdigits = int(np.log10(self.total)) + 1
        output += ('%' + str(numdigits) + 'd/%d') % (self.index, self.total)

        # Add fraction as percentage
        output += (' - {}%'.format(int(fraction*100)))

        self.index += 1
        
        sys.stdout.write('\b' * 200)
        sys.stdout.write('\r')
        sys.stdout.write(output)
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

