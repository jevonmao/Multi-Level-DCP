import torch
import os
import numpy as np

'''
gt poses can be extracted from numpy files. 
'''

def groundtruthLoading(path):

    
    files=os.listdir(path)
    gr_truths=torch.zeros(4,4,len(files))
    for file in files:
        full_path=os.path.join(path,file)
        res=np.load(full_path)
        gr_truths.append(res)
    
    return gr_truths

def DcpLoss(path,predictions):
    ground_truths=groundtruthLoading(path)
    pass


path='/scratch/fsa4859/deepmapping_pcr/D:\KITTI_odom\dataset_velodyne/dataset/icp_10'
DcpLoss()


    
    




