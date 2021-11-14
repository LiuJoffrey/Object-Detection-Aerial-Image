import torch
from base_predictor import BasePredictor
import torch.nn.functional as F
from models import Yolov1_vgg16bn
from YoloLoss import *


class ExamplePredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, dropout_rate=0.2, loss='YoloLoss', margin=0, threshold=None, **kwargs):
        super(ExamplePredictor, self).__init__(**kwargs)
        
        self.model = Yolov1_vgg16bn(pretrained=True)
        
        # use cuda
        self.model = self.model.to(self.device)
        # make optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        

        self.loss = {
            'YoloLoss': YoloLoss(7,2,5,0.5)
        }[loss]

    def _run_iter(self, batch, training):
        images = batch[0].cuda()
        target = batch[1].cuda()
        pred = self.model(images)

        #print(pred.size())
        #print(pred)
        
        loss = self.loss(pred,target)
        #print(loss)
        #exit()
        return pred, loss 

    def _predict_batch(self, batch):

        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        speaker = batch['speaker'].to(self.device).float()
        context = torch.cat((context, speaker), 2)

        logits, A_Matrix = self.model.forward(
            context.to(self.device),
            options.to(self.device))
        #F.sigmoid(logits)
        return logits
