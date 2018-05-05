import numpy as  np

class MLTools:
    def __init__(self):
        self.hello="hello"

    def vectorize_sequences(sequences, dimention=10000):
         results=np.zeros((len(sequences),dimention))
         for i,sequence in enumerate(sequences):
            results[i,sequence]=1.
         return results
