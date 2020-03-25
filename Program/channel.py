import numpy as np
from abc import *


class ChannelInterface (ABC):

    @abstractmethod
    # probability distribution
    def W (self, x, y):
        raise NotImplementedError ()

    @abstractmethod
    # sampling
    def sampling (self, x):
        raise NotImplementedError ()



class GaussianChannel (ChannelInterface):

    def __init__ (self, sigma):
        self.sigma = sigma

    # override
    def W (self, y, x):
        return np.exp (-(x-y)**2/(2*self.sigma**2)) / np.sqrt(2*self.sigma**2);

    # override
    def sampling (self, x):
        return np.random.normal (x, self.sigma);
