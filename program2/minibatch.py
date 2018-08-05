import numpy as np
import random

class minibatcher:
    def __init__(self, x, y, size):
        self.x_set = np.copy(x)
        self.y_set = np.copy(y)
        self.itemCount = x.shape[0]
        self.batchSize = size
        self.current = 0
        self.shuffle()

    def shuffle(self):
        self.current = 0;
        for i in range(self.itemCount):
            swapIn = random.randrange(i, self.itemCount)
            temp1 = self.x_set[i]
            temp2 = self.y_set[i]
            self.x_set[i] = self.x_set[swapIn]
            self.y_set[i] = self.y_set[swapIn]
            self.x_set[swapIn] = temp1
            self.y_set[swapIn] = temp2


    def next(self):
        if self.current >= self.itemCount:
            return None
        else:
            temp = self.current
            self.current = min(self.itemCount, self.current + self.batchSize)
            return [self.x_set[temp : self.current], self.y_set[temp : self.current]]
