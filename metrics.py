import torch
import numpy as np

class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    
    def __init__(self, at=10):
        self.at = at
        self.n = 0
        self.n_correct = 0
        self.name = 'Recall@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, predicts, batch, at=10):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        # TODO
        # This method will be called for each batch.
        # You need to
        # - increase self.n, which implies the total number of samples.
        # - increase self.n_corrects based on the prediction and labels
        #   of the batch.
        #print(predicts.shape)
        
        labels = batch['labels']
        if at == 1:
            for i in range(predicts.shape[0]):
                #v, most_at = predicts[i].max(0)
                most_at = torch.topk(predicts[i], k=at, dim=0)[1]
                #most_at = most_at.item()
                #print("most_at: ", most_at)
                ind = np.where(labels[i]==1)[0]
                #print("ind: ", ind)
                self.n+=1
                for l in ind:
                    if l in most_at:
                        self.n_corrects+=1
                    
        else:
            for i in range(predicts.shape[0]):
                ind = np.where(labels[i]==1)[0]
                most_at = torch.topk(predicts[i], k=at, dim=0)[1]
                self.n+=1
                for l in ind:
                    if l in most_at:
                        self.n_corrects+=1


        """
        predicts = (predicts>0.5).float()
        correct = (predicts == batch['labels'].float()).float().sum().item()
        #print(correct)
        self.n_corrects += correct
        self.n += (predicts.shape[0]*predicts.shape[1])
        """

    def get_score(self):
        return self.n_corrects / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.2f}'.format(score)
