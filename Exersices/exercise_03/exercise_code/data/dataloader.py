"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    @staticmethod
    def generate_chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def __iter__(self):

        if(self.shuffle):
            datasetIndices = np.random.permutation(len(self.dataset))
        else:
            datasetIndices = np.arange(0, len(self.dataset))

        chunks = list(self.generate_chunks(datasetIndices, self.batch_size))
        batches = []
        if(self.drop_last):
            if(len(self.dataset) % self.batch_size != 0):
                chunks.pop()

        for batch in chunks:
            newBatchArr = []
            for i in batch:
                newBatchArr.append(self.dataset[i].get("data"))
            batch = {"data": newBatchArr}
            batches.append(batch)
        
        return iter(batches)

    def __len__(self):
        length = None
        
        length = int(len(self.dataset) / self.batch_size)
        if(self.drop_last):
            return length
        if(len(self.dataset)%self.batch_size != 0):
            length += 1
        pass
        return length
