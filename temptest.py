import pandas as pd
import numpy as np
import pylab as pl

data = pd.read_table('kam2allinfo/iot-S35-L.txt')
pl.plot(data.y)
pl.plot(0.00005 * data.RLGYz)
pl.show()