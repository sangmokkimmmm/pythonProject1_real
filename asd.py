import tensorflow
import keras
import pandas as pd
import numpy as np

np.random.seed(0)

df = pd.DataFrame(np.random.randint(-100, 100, (20, 4)), columns=list('ABCD'))



def minmax_norm(df_input):
    return (df - df.min()) / (df.max() - df.min())

df_min_max_norm = minmax_norm(df)

print(df_min_max_norm)