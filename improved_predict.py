import torch
from torch.autograd import Variable
import torch.nn as nn
import os
from improved_models import Yolov1_vgg16bn, Yolov1_Resnet
import torchvision.transforms as transforms
import cv2
import numpy as np
import sys

all_label_name = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", 
"basketball-court","ground-track-field", "harbor", "bridge", "small-vehicle", 
"large-vehicle", "helicopter","roundabout", "soccer-ball-field", "swimming-pool",
 "container-crane"]


def decoder(pred):
    # pred 7*7*26

    grid_num = 14
    bbox_num = 2
    class_index = []
    boxes = []
    probs = []
    cell_size = 1./ grid_num 
    pred = pred.data
    pred = pred.squeeze(0) # drop batch size

    
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    mask1 = contain>0.1 # threshold to represent the bbox has obj or not
    mask2 = (contain == contain.max()) # choose the max score bbox whatever its score > "threshold" or not
    mask = (mask1+mask2).gt(0)

    

    for i in range(grid_num): # y cell
        for j in range(grid_num): # x cell
            #print(i, j)
            for b in range(bbox_num): # bbox in grid y x 
                #print(mask[i,j,b], pred[i,j,b*5+4])

                if mask[i,j,b] > 0:
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i]) * cell_size # to get the left-top cell 
                    box[:2] = box[:2]*cell_size+xy # to relative to image location

                    box_xyxy = torch.FloatTensor(box.size())
                    box_xyxy.zero_()

                    box_xyxy[:2] = box[:2] - 0.5*box[2:]
                    box_xyxy[2:] = box[:2] + 0.5*box[2:]
                    obj_max_prob, obj_max_index = torch.max(pred[i,j,10:], 0)
                    
                    
                    if (obj_max_prob*contain_prob) > 0.01: ## threshold to make sure that is an object # 0.1
                        boxes.append(box_xyxy.view(-1,4)) # to cat all obj together
                        class_index.append(torch.LongTensor([obj_max_index]))
                        probs.append(obj_max_prob*contain_prob)
    
    if len(boxes) == 0:
        #print("No object")
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        class_index = torch.zeros(1)
    else:
        
        boxes = torch.cat(boxes, 0)
        probs = torch.cat(probs)
        class_index = torch.cat(class_index)
    
    keep = nms(boxes, probs, class_index)
    
    return boxes[keep],class_index[keep],probs[keep]


def nms(boxes, probs, class_index, threshold=0.4):
    num_class = 16
    
    keep = []
    for c in range(num_class):

        current_class_index = []
        
        for ind_c in range(len(class_index)):
            if class_index[ind_c].item() == c:
                current_class_index.append(ind_c)
        current_class_index = np.array(current_class_index)
        
        current_prod = probs[current_class_index]
        

        if len(current_class_index)>0:
            

            _,sort_index = current_prod.sort(0,descending=True)
            #print(sort_index)
            

            #print(current_class_index[sort_index[0]])
            current_class_index = current_class_index[sort_index.numpy()] # current_class_index:  is the index of the current class 
            #print(current_class_index)
            #return current_class_index
            current_prod = probs[current_class_index]
            current_boxes = boxes[current_class_index]
            current_class = class_index[current_class_index]
            #print(current_prod)
            #print(current_boxes)
            #print(current_class)
            #print(len(current_class_index))
            
            while len(current_class_index)>0:

                x1 = current_boxes[:, 0]
                y1 = current_boxes[:, 1]
                x2 = current_boxes[:, 2]
                y2 = current_boxes[:, 3]
                areas = (x2-x1)*(y2-y1)

                
                i = current_class_index[0]
                
                keep.append(i.item())
                if len(current_class_index) == 1:
                    break
                
                xx1 = x1[1:].clamp(min=x1[0])
                yy1 = y1[1:].clamp(min=y1[0])
                xx2 = x2[1:].clamp(max=x2[0])
                yy2 = y2[1:].clamp(max=y2[0])
                
                
                
                w = (xx2-xx1).clamp(min=0)
                h = (yy2-yy1).clamp(min=0)
                inter = w*h
                
                iou = inter / (areas[0]+areas[1:]-inter)
                
                

                index = (iou<=threshold).nonzero().squeeze(-1)
                if len(index) == 0:
                    break
                

                current_class_index = current_class_index[index+1]
                if type(current_class_index) == np.int64:
                    current_class_index = np.array([current_class_index])
                
                
                current_boxes = current_boxes[index+1]
                
    
    return keep


def predict_gpu(model, image_name, root_path=""):
    image = cv2.imread(root_path+image_name)
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    with torch.no_grad():
        img = Variable(img[None,:,:,:]) # without backward gradient compute
        img = img.cuda()
        pred = model(img) # 1 7 7 26
        pred = pred.cpu()

    boxes,cls_indexs,probs =  decoder(pred)

    result = []
    pred_file = []
    for i, box in enumerate(boxes):
        x1 = int(box[0]*w)
        y1 = int(box[1]*h)
        x2 = int(box[2]*w)
        y2 = int(box[3]*h)

        cls_index = int(cls_indexs[i])
        prob = float(probs[i])

        result.append([(x1, y1), (x2, y2), all_label_name[cls_index], prob])
        pred_file.append(str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y1)+" "+str(x2)+" "+str(y2)+" "+str(x1)+" "+str(y2)+" "+all_label_name[cls_index]+" "+str(prob)+"\n")
        
    
    
    return result, pred_file

def write_txt(pred_file, img_name, output_path):
    path = os.path.join(output_path, img_name+".txt")
    with open(path, 'w') as f:
        for line in pred_file:
            f.write(line)


if __name__ == '__main__':

    model = Yolov1_Resnet(pretrained=False)
    path = "./models/A_improve_yolo.pkl.82"
    ckp = torch.load(path)    
    model.load_state_dict(ckp['model'])
    model.cuda()
    model.eval()
    
    arg = sys.argv
    val_img_path = arg[1]
    output_path = arg[2]

    #val_img_path = "../hw2_train_val/val1500/images/"
    img_file = os.listdir(val_img_path)
    print("In raw predict")

    for i in range(len(img_file)):
        image_name = os.path.join(val_img_path, img_file[i])

        #image_name = "./hw2_train_val/val1500/images/0000.jpg" # "./hw2_train_val/train15000/images/00001.jpg"
        image = cv2.imread(image_name)
        
        results, pred_file = predict_gpu(model,image_name)

        #print(pred_file)
        write_txt(pred_file, img_file[i][:-4], output_path)


    exit()
    """
    print(results)

    for r in results:
        left_top = r[0]
        right_bottom = r[1]
        class_name = r[2]
        prob = r[3]
        cv2.rectangle(image,left_top,right_bottom,(0,0,255),2)
    
    cv2.imshow("x", image)
    cv2.waitKey(0)
    """


    

    
