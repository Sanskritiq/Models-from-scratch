import numpy as np
import pandas

class Utily:
    @staticmethod
    def load(filename):
        df = pandas.read_csv(filename,sep='\t')
        keys = df.keys()
        return (np.array(df[keys[-2]]), np.array(df[keys[-1]]))

if __name__ == '__main__':
     data_x, data_y = Utily.load("data.tsv")