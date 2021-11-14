import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B 
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    
    def compute_iou(self, box1, box2): ## need to change 
               
        N = box1.size()[0]
        M = box2.size()[0]

        box1_wh = box1[:, 2:4] - box1[:, :2]
        box2_wh = box2[:, 2:4] - box2[:, :2]
        
        x1 = torch.max(box1[:,0], box2[:,0])
        y1 = torch.max(box1[:,1], box2[:,1]) # left-top
        x2 = torch.min(box1[:,2], box2[:,2])
        y2 = torch.min(box1[:,3], box2[:,3]) # right-bottom
        
        w = torch.max(x2-x1, torch.zeros_like(x2-x1))
        h = torch.max(y2-y1, torch.zeros_like(y2-y1))
        
        inter = w*h
        #print(inter)
        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1]) 
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1]) 

        iou = inter/(area1+area2-inter)
        
        return iou
    

        
    def forward(self, pred_tensor, target_tensor):

        N = pred_tensor.size()[0] # batch size

        ### To get the grid cell that contain objects ###
        coo_mask = target_tensor[:,:,:,4]>0 
        noo_mask = target_tensor[:,:,:,4]==0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)


        coo_pred = pred_tensor[coo_mask].view(-1,26)
        box_pred = coo_pred[:, :10].contiguous().view(-1,5) #coo_pred[:, :10].view(-1,5) # 2 BBox (x,y,w,h,c)
        class_pred = coo_pred[:, 10:]

        coo_target = target_tensor[coo_mask].view(-1,26)
        box_target = coo_target[:, :10].contiguous().view(-1,5) #coo_class[:, :10].view(-1,5) # 2 BBox (x,y,w,h,c) contiguous is to relocate the tensor to be contiguous
        class_target = coo_target[:, 10:]

        ### compute not obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,26)
        noo_target = target_tensor[noo_mask].view(-1,26)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4]=1
        noo_pred_mask[:,9]=1
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred only compute confidence loss size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,reduction='sum') # sum of the mse error for each element

        ### compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        #print(box_target.size()[0])
        for i in range(0,box_target.size()[0],2):
            box1 = box_pred[i:i+2]           
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:, :2] = box1[:, :2]/self.S - 0.5*box1[:, 2:4] # /self.S
            box1_xyxy[:, 2:4] = box1[:, :2]/self.S + 0.5*box1[:, 2:4]
            
            
            box2 = box_target[i:i+2]
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2]/self.S - 0.5*box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2]/self.S + 0.5*box2[:, 2:4]
            
            iou = self.compute_iou(box1_xyxy[:,:4], box2_xyxy[:,:4])
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1 # cause only 2 bbox,(0,1)
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()

        box_target_iou = Variable(box_target_iou).cuda()
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)

        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],reduction='sum') + \
                    F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),reduction='sum')
        
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        
        box_target_not_response[:,4]= 0
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],reduction='sum')
        
        class_loss = F.mse_loss(class_pred,class_target,reduction='sum')
            
        total_loss = (self.l_coord*loc_loss + contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N #2

        
        return total_loss

        



"""
def compute_iou(self, box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
        box1: (tensor) bounding boxes, sized [N,4].
        box2: (tensor) bounding boxes, sized [M,4].
    Return:
        (tensor) iou, sized [N,M].
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh<0] = 0  # clip at 0
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    print(iou)
    return iou
"""
