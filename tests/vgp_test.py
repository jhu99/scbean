import scbean.model.vgp as vgp
import pandas as pd
import multiprocessing as mp

# load data (example: MOB)
filepath = 'data/real_data/Rep11_MOB_count_matrix-1.tsv'
data = pd.read_csv(filepath, sep='\t')

# data preprocessing
position = pd.DataFrame(index=data.index)
position['x'] = data['Unnamed: 0'].str.split('x').str.get(0).map(float)
position['y'] = data['Unnamed: 0'].str.split('x').str.get(1).map(float)
data.drop('Unnamed: 0', axis=1, inplace=True)
data = data.T[data.sum(0) >= 10].T  # Filter practically unobserved genes
data = data.T  # genes * position
position = position[data.sum(0) >= 10]
data = data.T[data.sum(0) >= 10].T

X = position.values
Y = data

if __name__ == '__main__':
    result = vgp.run(X, Y, mp.cpu_count())
    print(result)
