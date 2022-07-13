import torch
import torch.nn as nn

from pointconv_util_kitti import (All2AllCostVolume, Conv1d, CostVolume,
                            FlowPredictor, PointnetFpModule, PointNetSaModule,
                            SetUpconvModule, WarpingLayers,
                            index_points_gather)

scale = 1.0


class ThreeDFlow_Kitti(nn.Module):
    def __init__(self,  is_training, bn_decay=None):
        super(ThreeDFlow_Kitti, self).__init__()
        RADIUS1 = 0.5
        RADIUS2 = 1.0
        RADIUS3 = 2.0
        RADIUS4 = 4.0

        self.layer0 = PointNetSaModule(npoint=4096, radius=RADIUS1, nsample=64, in_channels=3,mlp=[16,16,32],mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, use_fps=False)
        self.layer1 = PointNetSaModule(  npoint=1024, radius=RADIUS1, nsample=64, in_channels=32,mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer2 = PointNetSaModule(  npoint=256, radius=RADIUS2, nsample=16, in_channels=64,mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer3_2 = PointNetSaModule(  npoint=64, radius=RADIUS3, nsample=16,in_channels=128,mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)



        self.cost1 = All2AllCostVolume(radius=None, nsample=4, nsample_q=256, in_channels=128,mlp1=[256,128,128], mlp2 = [256,128], is_training=is_training, bn_decay=bn_decay, bn=True, pooling='max', knn=True, corr_func='concat')

        self.layer3_1 = PointNetSaModule(   npoint=64, radius=RADIUS3, nsample=8,in_channels=128, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer4_1 = PointNetSaModule(  npoint=16, radius=RADIUS4, nsample=8,in_channels=256, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)

        self.upconv1 = SetUpconvModule(  nsample=8, radius=2.4,in_channels=[256,512],mlp=[256,256,512], mlp2=[512],is_training=is_training, bn_decay=bn_decay, knn=True)


        # Layer3
        self.conv1 = Conv1d(512,3)

        self.warping1 = WarpingLayers()

        self.cost2 = CostVolume(radius=None, nsample=4, nsample_q=6, in_channels=256,mlp1=[512,256,256], mlp2=[512,256], is_training=is_training, bn_decay=bn_decay,bn=True, pooling='max', knn=True, corr_func='concat')

        self.flow_pred1 = FlowPredictor([256,512,256,3], mlp=[512,256,256], is_training = is_training , bn_decay = bn_decay,npoint=64)

        self.conv2 = Conv1d(256,3)


        # Layer 2
        self.upconv2 = SetUpconvModule(  nsample=8, radius=1.2, in_channels=[128,256],mlp=[256,128,128], mlp2=[128],is_training=is_training, bn_decay=bn_decay, knn=True)

        self.fp1 = PointnetFpModule(in_channels=3,mlp=[], is_training=is_training, bn_decay=bn_decay)

        self.warping2 = WarpingLayers()

        self.cost3 = CostVolume(radius=None, nsample=4, nsample_q=6, in_channels=128,mlp1=[256,128,128], mlp2=[256,128], is_training=is_training, bn_decay=bn_decay, bn=True, pooling='max', knn=True, corr_func='concat')

        self.flow_pred2 = FlowPredictor([128,128,128,3],mlp=[256,128,128], is_training = is_training , bn_decay = bn_decay ,npoint=256)

        self.conv3 = Conv1d(128,3)


        #Layer 1
        self.upconv3 = SetUpconvModule(  nsample=8, radius=1.2,in_channels=[64,128], mlp=[256,128,128], mlp2=[128],is_training=is_training, bn_decay=bn_decay, knn=True)

        self.fp2 = PointnetFpModule(in_channels=3,mlp=[],is_training=is_training,bn_decay=bn_decay)

        self.warping3 = WarpingLayers()

        self.cost4 = CostVolume(radius=None, nsample=4, nsample_q=6, in_channels=64,mlp1=[128,64,64], mlp2=[128,64],is_training=is_training, bn_decay=bn_decay,bn=True, pooling='max', knn=True, corr_func='concat')

        self.flow_pred3 = FlowPredictor([64,128,64,3],mlp=[256,128,128], is_training = is_training , bn_decay = bn_decay ,npoint=1024)

        self.conv4 = Conv1d(128,3)


        #layer 0 
        self.upconv4 = SetUpconvModule(  nsample=8, radius=1.2,in_channels=[32,128], mlp=[128,64,64], mlp2=[64],is_training=is_training, bn_decay=bn_decay, knn=True)
        
        self.warping4 = WarpingLayers()
        
        self.cost5 = CostVolume(radius=None, nsample=4, nsample_q=6, in_channels=32,mlp1=[64,32,32], mlp2=[64,32],is_training=is_training, bn_decay=bn_decay,bn=True, pooling='max', knn=True, corr_func='concat')
        
        self.fp3 = PointnetFpModule( in_channels=3,mlp=[], is_training=is_training, bn_decay=bn_decay)
        
        self.flow_pred4 = FlowPredictor([32,64,32,3],mlp=[128,64,64], is_training = is_training , bn_decay = bn_decay, npoint=4096)
        
        self.conv5 = Conv1d(64,3)
        
        


    def forward(self, xyz1,xyz2,color1,color2,label,mask):
        # xyz1, xyz2: B, N, 3
        # color1, color2: B, N, 3
        # label: B,N,3
    
        l0_xyz_f1_raw = xyz1
        l0_xyz_f2_raw = xyz2

        xyz1_center = torch.mean(xyz1,dim=1,keepdim=True)   # (b,1,3)

        xyz1 = xyz1 - xyz1_center   # (b,n,3)
        xyz2 = xyz2 - xyz1_center   # (b,n,3)

        l0_xyz_f1 = xyz1

        l0_points_f1 = color1

        if label is None:
            label = torch.zeros(xyz1.size(),device='cuda')

        l0_label_f1 = label
        # l0_mask_f1 = mask

        l0_xyz_f2 = xyz2

        l0_points_f2 = color2

       
        l0_xyz_f1, l0_label_f1, l0_points_f1, pc1_sample = self.layer0(l0_xyz_f1, l0_xyz_f1_raw, l0_label_f1, l0_points_f1)  #(b,2048,3) (b,2048,3) (b,2048,32)
        
        l0_mask = torch.split(mask,4096,1)[0] # 2048

        l1_xyz_f1, l1_label, l1_points_f1,idx_1024 = self.layer1(l0_xyz_f1, None, l0_label_f1, l0_points_f1)  #(b,1024,3) (b,1024,3) (b,1024,64)
        
        l1_mask = index_points_gather(l0_mask, idx_1024) # 1024
        
        l2_xyz_f1, l2_label, l2_points_f1,idx_256 = self.layer2(l1_xyz_f1, None, l1_label, l1_points_f1)    #(b,256,3) (b,256,3) (b,256,128)

        l2_mask = index_points_gather(l1_mask, idx_256) # 256


        
        l0_xyz_f2, _, l0_points_f2, pc2_sample = self.layer0(l0_xyz_f2, l0_xyz_f2_raw, label, l0_points_f2)  #(b,2048,3) (b,2048,3) (b,2048,32)

        l1_xyz_f2, _, l1_points_f2,_ = self.layer1(l0_xyz_f2, None, l0_label_f1, l0_points_f2)  #(b,1024,3) (b,1024,3) (b,1024,64)

        l2_xyz_f2, _, l2_points_f2,_ = self.layer2(l1_xyz_f2, None, l1_label, l1_points_f2)   #(b,256,3) (b,256,3) (b,256,128)

        l3_xyz_f2, _, l3_points_f2,_ = self.layer3_2(l2_xyz_f2, None, l2_label, l2_points_f2)  #(b,64,3) (b,64,3) (b,64,256)



        l2_points_f1_new = self.cost1(l2_xyz_f1, l2_points_f1, l2_xyz_f2, l2_points_f2)  # (b,256,128)

        l3_xyz_f1, l3_label, l3_points_f1,idx_64 = self.layer3_1(l2_xyz_f1, None, l2_label, l2_points_f1_new) # (b,64,3) (b,64,3) (b,64,256)
        
        l3_mask = index_points_gather(l2_mask, idx_64) # 64

        l4_xyz_f1, _, l4_points_f1,_ = self.layer4_1(l3_xyz_f1, None, l3_label, l3_points_f1)  #(b,16,3) (b,16,3) (b,16,512)

        l3_feat_f1 = self.upconv1(l3_xyz_f1, l4_xyz_f1, l3_points_f1, l4_points_f1)  #(b,64,512)


        #Layer 3
        l3_points_f1_new = l3_feat_f1  #(b,64,512)

        l3_flow_coarse = self.conv1(l3_points_f1_new)  #(b,64,3)

        l3_flow_warped = self.warping1(l3_xyz_f1, l3_flow_coarse)   #(b,64,3)

        l3_cost_volume = self.cost2(l3_flow_warped, l3_points_f1, l3_xyz_f2, l3_points_f2)  #(b,64,256)

        l3_flow_finer = self.flow_pred1(l3_points_f1, l3_points_f1_new, l3_cost_volume,l3_flow_coarse,pc = l3_xyz_f1)  # (b,64,256)

        l3_flow_det = self.conv2(l3_flow_finer) #(b,64,3)

        l3_flow = l3_flow_coarse + l3_flow_det  #(b,64,3)

        #Layer 2
        l2_points_f1_new = self.upconv2(l2_xyz_f1, l3_xyz_f1, l2_points_f1, l3_flow_finer)  #(b,256,128)

        l2_flow_coarse = self.fp1(l2_xyz_f1, l3_xyz_f1, None, l3_flow)  #(b,256,3)

        l2_flow_warped = self.warping2(l2_xyz_f1, l2_flow_coarse)  #(b,256,3)

        l2_cost_volume = self.cost3(l2_flow_warped, l2_points_f1, l2_xyz_f2, l2_points_f2)  #(b,256,128)

        l2_flow_finer = self.flow_pred2(l2_points_f1, l2_points_f1_new, l2_cost_volume,l2_flow_coarse,pc = l2_xyz_f1)   #(b,256,128)

        l2_flow_det = self.conv3(l2_flow_finer)    #(b,256,3)

        l2_flow = l2_flow_coarse + l2_flow_det    #(b,256,3)

        #Layer 1
        l1_points_f1_new = self.upconv3(l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_flow_finer)   #(b,1024,128)

        l1_flow_coarse = self.fp2(l1_xyz_f1, l2_xyz_f1, None, l2_flow)  #(b,1024,3)

        l1_flow_warped = self.warping3(l1_xyz_f1, l1_flow_coarse)  #(b,1024,3)

        l1_cost_volume = self.cost4(l1_flow_warped, l1_points_f1, l1_xyz_f2, l1_points_f2)   #(b,1024,64)

        l1_flow_finer = self.flow_pred3(l1_points_f1, l1_points_f1_new, l1_cost_volume,l1_flow_coarse, pc=l1_xyz_f1)  #(b,1024,128)

        l1_flow_det = self.conv4(l1_flow_finer)     #(b,1024,3)

        l1_flow = l1_flow_coarse + l1_flow_det  #(b,1024,3)


        #Layer 0
        l0_points_f1_new = self.upconv4(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l1_flow_finer)   #(b,2048,64)

        l0_flow_coarse = self.fp3(l0_xyz_f1, l1_xyz_f1, None, l1_flow) #(b,2048,3)
        
        l0_flow_warped = self.warping4(l0_xyz_f1, l0_flow_coarse)  #(b,2048,3)
        
        l0_cost_volume = self.cost5(l0_flow_warped, l0_points_f1, l0_xyz_f2, l0_points_f2)   #(b,2048,32)
     
        l0_flow_finer = self.flow_pred4(l0_points_f1, l0_points_f1_new, l0_cost_volume,l0_flow_coarse, pc=l0_xyz_f1)  #(b,2048,64)
        
        l0_flow_det = self.conv5(l0_flow_finer)     #(b,2048,3)

        l0_flow = l0_flow_coarse + l0_flow_det  #(b,2048,3)


        l0_flow = l0_flow.permute(0,2,1) #(b,3,2048)
        l1_flow = l1_flow.permute(0,2,1)  #(b,3,1024)
        l2_flow = l2_flow.permute(0,2,1)  #(b,3,256)
        l3_flow = l3_flow.permute(0,2,1)   #(b,3,64)
        l0_label_f1 = l0_label_f1.permute(0,2,1)
        l1_label = l1_label.permute(0,2,1)
        l2_label = l2_label.permute(0,2,1)
        l3_label = l3_label.permute(0,2,1)
        l0_xyz_f1 = l0_xyz_f1.permute(0,2,1)
        l1_xyz_f1 = l1_xyz_f1.permute(0,2,1)
        l2_xyz_f1 = l2_xyz_f1.permute(0,2,1)
        l3_xyz_f1 = l3_xyz_f1.permute(0,2,1)
        l0_xyz_f2 = l0_xyz_f2.permute(0, 2, 1)
        l1_xyz_f2 = l1_xyz_f2.permute(0, 2, 1)
        l2_xyz_f2 = l2_xyz_f2.permute(0, 2, 1)
        l3_xyz_f2 = l3_xyz_f2.permute(0, 2, 1)
        flow = [l0_flow, l1_flow, l2_flow, l3_flow]
        label = [l0_label_f1, l1_label, l2_label, l3_label]
        pc1 = [l0_xyz_f1, l1_xyz_f1, l2_xyz_f1, l3_xyz_f1]
        pc2 = [l0_xyz_f2, l1_xyz_f2, l2_xyz_f2, l3_xyz_f2]
        mask_list = [l0_mask,l1_mask,l2_mask,l3_mask]

        return flow,label,pc1,pc2,pc1_sample,pc2_sample,mask_list

def multiScaleLoss(pred_flows, gt_flows, mask, alpha = [0.02, 0.04, 0.08, 0.16]):

    num_scale = len(pred_flows)

    total_loss = torch.zeros(1).cuda()

    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i].permute(0, 2, 1)
        total_loss += alpha[i] * (torch.norm(diff_flow, dim = 2,keepdim=True)*mask[i]).sum(dim = 1).mean()

    return total_loss


