import cv2
import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.misc import imread, imresize
import random


train_file = "hw2_train_val/train15000"
train_file_img = "./hw2_train_val/train15000/images"
train_file_ann = "./hw2_train_val/train15000/labelTxt_hbb"
valid_file = "hw2_train_val/val1500"

all_label_name = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", 
"basketball-court","ground-track-field", "harbor", "bridge", "small-vehicle", 
"large-vehicle", "helicopter","roundabout", "soccer-ball-field", "swimming-pool",
 "container-crane"]

class Imgdataset(data.Dataset):
    def __init__(self, img_folder, ann_folder, img_size, train, S, B, C, transform):
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.img_size = img_size
        self.train = train
        self.S = S
        self.B = B
        self.C = C
        self.transforms = transform
        self.bboxes = []
        self.labels = []
        self.difficulty = []
        self.all_img_path = []

        all_ann_path = []
        
        all_ann_file = os.listdir(ann_folder)
        for file_name in all_ann_file:
            all_ann_path.append(os.path.join(ann_folder, file_name))
            self.all_img_path.append(os.path.join(img_folder, file_name[:-3]+"jpg"))
        
        for i in range(len(all_ann_path))[:]:
            with open(all_ann_path[i]) as f:
                lines = f.readlines()

            bbox = []
            label = []
            difficulty = []
            for line in lines:
                splited = line.strip().split()
                
                x1 = float(splited[0]) # xmin
                y1 = float(splited[1]) # ymin
                x2 = float(splited[4]) # xmax
                y2 = float(splited[5]) # ymax
                """
                x_center = (x1+x2)/2
                y_center = (y1+y2)/2
                w = x2-x1
                h = y2-y1
                """
                bbox.append([x1, y1, x2, y2])
                #bbox.append([x_center, y_center, w, h])
                label.append(all_label_name.index(splited[8])+1)
                difficulty.append(int(splited[9]))
            
            self.bboxes.append(torch.Tensor(bbox))
            self.labels.append(torch.LongTensor(label))
            self.difficulty.append(torch.LongTensor(difficulty))

    def __len__(self):
        return len(self.labels)   

    def __getitem__(self, index):
        #print(self.all_img_path[index], self.bboxes[index], self.labels[index], self.difficulty[index])
        #if index != 9:
        #    return ("no")
        img_path = self.all_img_path[index]
        #img_path = "./hw2_train_val/train15000/images/05343.jpg"
        
        #index = self.all_img_path.index(img_path)
        
        #print(img_path)
        img = cv2.imread(img_path)

        boxes = self.bboxes[index].clone()
        labels = self.labels[index].clone()
        difficulty = self.difficulty[index].clone()


        """
        f_img = img.copy()
        for i in range(len(boxes)):
                cv2.rectangle(f_img,(int(boxes[i][0]),int(boxes[i][1])),(int(boxes[i][2]),int(boxes[i][3])),(0,0,255), 2)
        cv2.imwrite("./test3/raw_"+str(index)+".jpg", f_img)
        """

        if self.train:
            
            img, boxes = self.random_flip(img, boxes)
            img,boxes = self.random_scale(img,boxes)
            img = self.random_blur(img)
            img = self.random_bright(img)
            img = self.random_hue(img)
            img = self.random_saturation(img)
            img = self.random_grayscale(img)
            
            img = self.random_noise(img)

            img,boxes,labels,difficulty = self.random_shift(img,boxes,labels,difficulty)
            #img,boxes,labels,difficulty = self.random_crop(img,boxes,labels,difficulty)

        
        
        """
        f_img = img.copy()
        for i in range(len(boxes)):
                cv2.rectangle(f_img,(int(boxes[i][0]),int(boxes[i][1])),(int(boxes[i][2]),int(boxes[i][3])),(0,0,255), 2)
        cv2.imwrite("./test3/test_"+str(index)+".jpg", f_img)
        """

        h, w, _ = img.shape

        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes) # normalized by raw image height and width

        img = self.BGR2RGB(img)
        img = cv2.resize(img,(self.img_size,self.img_size))

        target = self.encoder(boxes,labels)# 7x7x26
        for t in self.transforms:
            img = t(img)

        return img, target
        
        """    
        f_img = img.copy()
        for i in range(len(boxes)):
                cv2.rectangle(f_img,(int(boxes[i][0]),int(boxes[i][1])),(int(boxes[i][2]),int(boxes[i][3])),(0,0,255), 2)
        cv2.imwrite("./test/f_"+str(index)+".jpg", f_img)
        """
        
    def encoder(self, boxes, label):
        grid_num = self.S
        target = torch.zeros((grid_num,grid_num,self.B*5+self.C)) # (7,7,26)
        cell_size = 1./grid_num
        wh = boxes[:,2:]-boxes[:,:2] # (n_object, w, h)
        cxcy = (boxes[:,2:]+boxes[:,:2])/2 # (n_object, x_center, y_center)
        
        for i in range(cxcy.size()[0])[:]:
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 # mapping to the corresponding grid cell
            #print(cxcy_sample, ij)
            if target[int(ij[1]), int(ij[0]), 4] == 0: 
                target[int(ij[1]), int(ij[0]), 4] = 1
                target[int(ij[1]), int(ij[0]), 9] = 1
                target[int(ij[1]), int(ij[0]), label[i]+9] = 1 
                delta_xy = cxcy_sample/cell_size - ij
                target[int(ij[1]),int(ij[0]),2:4] = wh[i]
                target[int(ij[1]),int(ij[0]),:2] = delta_xy
                target[int(ij[1]),int(ij[0]),7:9] = wh[i]
                target[int(ij[1]),int(ij[0]),5:7] = delta_xy

        return target
            
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    def BGR2GRA(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    def random_flip(self, img, boxes):
        #cv2.imshow("r",img)
        
        if random.random() > 0.5:
            h,w,c = img.shape
            img = img[:,::-1, :]
            #cv2.imshow("m",img_mirror)
            #cv2.waitKey(0)
            
            x_left = w - boxes[:,0]
            x_right = w - boxes[:,2]
            
            boxes[:,0] = x_right
            boxes[:,2] = x_left

        if random.random() > 0.5:
            h,w,c = img.shape
            img = img[::-1,:, :]
            
            y_top = h - boxes[:,1]
            y_bottom = h - boxes[:,3]
            
            boxes[:,1] = y_bottom
            boxes[:,3] = y_top

        return img, boxes 

          

    def random_scale(self, img, boxes):
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            h, w, c = img.shape
            img = cv2.resize(img, (w, int(h*scale)))
            scale_tensor = torch.FloatTensor([[1,scale,1,scale]]).expand_as(boxes)
            boxes = boxes*scale_tensor
            boxes[:, 1] = torch.clamp(boxes[:, 1], 0, int(h*scale))
            boxes[:, 3] = torch.clamp(boxes[:, 3], 0, int(h*scale))
        
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            h, w, c = img.shape
            img = cv2.resize(img, (int(w*scale), h))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes*scale_tensor
            boxes[:, 0] = torch.clamp(boxes[:, 0], 0, int(w*scale))
            boxes[:, 2] = torch.clamp(boxes[:, 2], 0, int(w*scale))
        
        return img, boxes

    def random_blur(self, img):
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img,(5,5),0)
        return img

    def random_bright(self, img):
        
        if random.random() > 0.5:
            value = int(random.uniform(-10,10))
            hsv = self.BGR2HSV(img)
            #hsv = hsv.astype('float32')
            h, s, v = cv2.split(hsv)
            v = v.astype('float32')
            
            """
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
            """
            v = v+value
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            final_hsv = cv2.merge((h, s, v))
            #final_hsv = final_hsv.astype('uint8')
            img = self.HSV2BGR(final_hsv)
        
        return img

    def random_hue(self, img):
        if random.random()>0.5:
            hsv = self.BGR2HSV(img)
            #hsv = hsv.astype('float32')
            h, s, v = cv2.split(hsv)
            value = random.uniform(0.5,1.2)
            h = h.astype('float32')
            h = h*value
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            final_hsv = cv2.merge((h,s,v))
            #final_hsv = final_hsv.astype('uint8')
            img = self.HSV2BGR(final_hsv)
            
        return img

    def random_saturation(self, img):
        if random.random()>0.5:
            hsv = self.BGR2HSV(img)
            #hsv = hsv.astype('float32')
            h, s, v = cv2.split(hsv)
            value = random.uniform(0.5,1.2)
            s = s.astype('float32')
            s = s*value
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            final_hsv = cv2.merge((h,s,v))
            #final_hsv = final_hsv.astype('uint8')
            img = self.HSV2BGR(final_hsv)
            
        return img

    def random_grayscale(self, img):
        if random.random()>0.5:
            img = self.BGR2GRA(img)
            img = np.expand_dims(img, 2)
            img = np.concatenate((img,img,img), axis=2)
            return img
        return img

    def random_noise(self, img):
        if random.random()>0.5:
            h, w, c = img.shape
            mean = 0
            var = 3
            sigma = var**0.5
            gaussian = np.random.normal(mean, sigma, (h, w))
            noisy_image = np.zeros(img.shape, np.float32)
            noisy_image[:, :, 0] = img[:, :, 0] + gaussian
            noisy_image[:, :, 1] = img[:, :, 1] + gaussian
            noisy_image[:, :, 2] = img[:, :, 2] + gaussian
            cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            noisy_image = noisy_image.astype(np.uint8)
            return noisy_image

        if random.random()>0.5:
            output = np.zeros(img.shape,np.uint8)
            thres = 1 - 0.05 
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    rdn = random.random()
                    if rdn < 0.05:
                        output[i][j] = 0
                    elif rdn > 0.99:
                        output[i][j] = 255
                    else:
                        output[i][j] = img[i][j]
            return output

        return img

    def random_shift(self, img,boxes,labels,difficulty):
        
        if random.random()>0.5:
            h,w,_ = img.shape
            shift_img = np.zeros((img.shape), dtype=img.dtype) # should be the mean of the training image 
            ### Later 

            shift_x = int(random.uniform(-0.2*w, 0.2*w))
            shift_y = int(random.uniform(-0.2*h, 0.2*h))

            if shift_x >=0 and shift_y>=0:
                shift_img[shift_y:, shift_x:] = img[:h-shift_y, :w-shift_x]
            elif shift_x >=0 and shift_y<0:
                shift_img[:h-(-shift_y), shift_x:] = img[-shift_y:, :w-shift_x]
            elif shift_x < 0 and shift_y>=0:
                shift_img[shift_y:, :w-(-shift_x)] = img[:h-shift_y, -shift_x:]
            elif shift_x < 0 and shift_y<0:
                shift_img[:h-(-shift_y), :w-(-shift_x)] = img[-shift_y:, -shift_x:]

            center_xy = (boxes[:, :2]+boxes[:, 2:4])/2
            #print(center_xy)
            shift_xy = torch.FloatTensor([shift_x, shift_y])
            #print(shift_xy)
            center_xy = center_xy + shift_xy

            mask1 = (center_xy[:,0] > 0) & (center_xy[:,0] < w)
            mask2 = (center_xy[:,1] > 0) & (center_xy[:,0] < h)
            mask = (mask1 & mask2).view(-1,1)
            
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
                        
            if len(boxes_in) == 0:
                return img, boxes,labels,difficulty

            box_shift = torch.FloatTensor([shift_x,shift_y,shift_x,shift_y])
            boxes_in = boxes_in+box_shift
            boxes_in[:, 0] = torch.clamp(boxes_in[:, 0], 0, w)
            boxes_in[:, 2] = torch.clamp(boxes_in[:, 2], 0, w)
            boxes_in[:, 1] = torch.clamp(boxes_in[:, 1], 0, h)
            boxes_in[:, 3] = torch.clamp(boxes_in[:, 3], 0, h)
            labels_in = labels[mask.view(-1)]
            difficulty_in = difficulty[mask.view(-1)]
            #print(boxes_in)
            return shift_img, boxes_in, labels_in,difficulty_in

        return img, boxes,labels,difficulty

    def random_crop(self, img,boxes,labels,difficulty):
        if random.random()>0.5:
            h,w,_ = img.shape

            h_crop = random.uniform(0.6*h,h)
            w_crop = random.uniform(0.6*w,w)

            x_loc = random.uniform(0, w-w_crop)
            y_loc = random.uniform(0, h-h_crop)
            x_loc, y_loc, w_crop, h_crop = int(x_loc), int(y_loc), int(w_crop), int(h_crop)
            
            center_xy = (boxes[:, :2]+boxes[:, 2:4])/2

            #center_xy = center_xy + torch.FloatTensor([x_loc,y_loc])

            mask1 = (center_xy[:,0] > x_loc) & (center_xy[:,0] < x_loc+w_crop)
            mask2 = (center_xy[:,1] > y_loc) & (center_xy[:,1] < y_loc+h_crop)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)

            if len(boxes_in) == 0:
                return img, boxes,labels,difficulty
            
            box_shift = torch.FloatTensor([x_loc,y_loc,x_loc,y_loc])
            

            boxes_in = boxes_in - box_shift
            

            boxes_in[:, 0] = torch.clamp(boxes_in[:, 0], 0, w_crop)
            boxes_in[:, 2] = torch.clamp(boxes_in[:, 2], 0, w_crop)
            boxes_in[:, 1] = torch.clamp(boxes_in[:, 1], 0, h_crop)
            boxes_in[:, 3] = torch.clamp(boxes_in[:, 3], 0, h_crop)
            
            crop_img = img[y_loc:y_loc+h_crop,x_loc:x_loc+w_crop,:]
            labels_in = labels[mask.view(-1)]
            difficulty_in = difficulty[mask.view(-1)]
            
            return crop_img, boxes_in, labels_in,difficulty_in
        return img,boxes,labels,difficulty


            #shift_img = np.zeros((img.shape), dtype=img.dtype) # should be the mean of the training image 



    

def main():
    img_file = os.listdir(train_file_img)
    img = cv2.imread(os.path.join(train_file_img, img_file[0]))
    train_data = Imgdataset(train_file_img, train_file_ann, 448, True, S=7, B=2, C=16, transform=[transforms.ToTensor()])
    dataload = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

    for i, batch in enumerate(dataload):
        #print(i)
        if i > 20:
            break
        #print(batch[0].size(), batch[1].size())
        




if __name__ == '__main__':
    main()