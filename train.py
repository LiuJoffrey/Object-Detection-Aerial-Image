import os 
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from models import Yolov1_vgg16bn
from YoloLoss import *
from callbacks import *
from dataset import *


train_file = "hw2_train_val/train15000"
train_file_img = "./hw2_train_val/train15000/images"
train_file_ann = "./hw2_train_val/train15000/labelTxt_hbb"
valid_file = "hw2_train_val/val1500"
valid_file_img = "./hw2_train_val/val1500/images"
valid_file_ann = "./hw2_train_val/val1500/labelTxt_hbb"

all_label_name = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", 
"basketball-court","ground-track-field", "harbor", "bridge", "small-vehicle", 
"large-vehicle", "helicopter","roundabout", "soccer-ball-field", "swimming-pool",
 "container-crane"]


use_gpu = torch.cuda.is_available()

train_data = Imgdataset(train_file_img, train_file_ann, 448, True, S=7, B=2, C=16, transform=[transforms.ToTensor()])
valid_data = Imgdataset(valid_file_img, valid_file_ann, 448, False, S=7, B=2, C=16, transform=[transforms.ToTensor()])

from example_predictor import ExamplePredictor
PredictorClass = ExamplePredictor
predictor = PredictorClass(
        #metrics=[Recall(at=10)], #
        valid = valid_data,
        max_epochs=200,
        batch_size=32
    )

load_model = "./best_model/0_125_simple_yolo.pkl.119"#./checkpoint_4/simple_yolo.pkl.13
if load_model != "":
    print("Load pre-trained")
    predictor.load(load_model)

model_dir = "./checkpoint_9/"

model_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'simple_yolo.pkl'),
        'loss', 1, 'all')

metrics_logger = MetricsLogger(
    os.path.join(model_dir, 'log.json')
)

print("Start Training!")
predictor.fit_dataset(train_data, callbacks=[model_checkpoint, metrics_logger])