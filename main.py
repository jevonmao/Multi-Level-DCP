import os
import argparse
import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.metrics import mean_squared_error

from models import utils, loss2, deepmapping2
#from data_loader import kitti_data
from data_loader_kitti import kitti_loader_exp2
from data_loader_kitti import data_load_test
from lib.timer import AverageMeter
from eval_icp import icp_o3d
import logging
import open3d as o3d

# Code runs deterministically 
torch.backends.cudnn.deterministic = True
np.random.seed(69)
torch.manual_seed(69)

def eval_metric(rotation, translation, transformation_gt):

    k = transformation_gt.shape[0]
    success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
    

    for i in range(k):
       

        identity = np.eye(3)

        rre = mean_squared_error(np.transpose(rotation[i, :, :].reshape(3,3)) @ transformation_gt[i, :3, :3].reshape(3,3), identity)
        rte = mean_squared_error(translation[i, :].reshape(3, 1), transformation_gt[i, :3, 3].reshape(3, 1))

        rte_meter.update(rte)
        rre_meter.update(rre)

        if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
            success_meter.update(1)
        else:
            success_meter.update(0)
            logging.info(f"Failed with RTE: {rte}, RRE: {rre}")

    print('RTE:', rte_meter.avg)
    print('RRE:', rre_meter.avg)
    print('Success (%):', success_meter.avg*100)

    return rre_meter.avg, rte_meter.avg


if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='exp2_train_4',help='experiment name')
    parser.add_argument('-e','--n_epochs',type=int,default=40,help='number of epochs')
    parser.add_argument('-b','--batch_size',type=int,default=4,help='batch_size')
    parser.add_argument('-l','--loss',type=str,default='bce_ch',help='loss function')
    parser.add_argument('-n','--n_samples',type=int,default=45,help='number of sampled unoccupied points along rays')
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate') # default is 0.001
    #parser.add_argument('-d','--root',type=str,default='/mnt/NAS/home/xinhao/deepmapping/main/data/kitti/',help='root path')
    parser.add_argument('-d','--root',type=str,default='/scratch/fsa4859/deepmapping_pcr/data_loader_kitti/config/test_kitti.txt',help='root path')
    parser.add_argument('-t','--traj',type=str,default='2011_09_30_drive_0018_sync_tfvpr',help='trajectory file folder')
    parser.add_argument('-v','--voxel_size',type=float,default=1,help='size of downsampling voxel grid')
    parser.add_argument('-m','--model', type=str, default=None,help='pretrained model name')
    parser.add_argument('-i','--init', type=str, default=None,help='init pose')
    parser.add_argument('--log_interval',type=int,default=1,help='logging interval of saving results')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--n_transf_layers',type=int,default=3,metavar='N',
                        help="Number of transformer layers")
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_keypoints', type=int, default=600, metavar='N',
                        help='Number of key poinits')
    parser.add_argument('--des_dim', type=int, default=256, metavar='N',
                        help='Neiborhood descriptor dimension')
    parser.add_argument('--k', type=int, default=4, metavar='N',
                        help='No. of nearest neighbors')
    parser.add_argument('--dim', type=int, default=16, metavar='N',
                        help='Dim')
    opt = parser.parse_args()
    checkpoint_dir = os.path.join('./results/', opt.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save parser arguments
    utils.save_opt(checkpoint_dir, opt)
    device = torch.device("cuda:2")
    print("Device:", device)

    print('loading dataset........')
    #train_dataset = kitti_data.Kitti('D:\kitti_group', opt.traj, opt.voxel_size, init_pose=None, 
            #group=True, group_size=9)
    #train_loader = DataLoader(train_dataset, batch_size=4, num_workers=8)
    dataset = kitti_loader_exp2.KITTINMPairDataset(phase='train',
                                                   random_rotation=False,
                                                   random_scale=False,)
    train_loader = DataLoader(dataset,
                              batch_size=opt.batch_size,
                              num_workers=8,
                              shuffle=False)

    #test_dataset = data_load_test.KITTI('/mnt/NAS/home/xinhao/deepmapping/main/data/kitti/', opt.traj, opt.voxel_size, init_pose=None, 
            #group=True, group_size=9)
    test_dataset = kitti_loader_exp2.KITTINMPairDataset(phase='test')
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=8)
    
    #test_loader =  DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)
    loss_fn = eval('loss2.'+opt.loss)
    print('creating model......')

    model = deepmapping2.DeepMappingKITTI(loss_fn=loss_fn, args=opt,n_samples=opt.n_samples).to(device)
    model=model.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    lr_step = [25, 50, 75]
    scheduler = MultiStepLR(optimizer,
                                milestones=[int(i) for i in lr_step],
                                gamma=0.3)

    if opt.model is not None:
        utils.load_checkpoint(opt.model, model, optimizer)

    print('start training')
    torch.cuda.empty_cache()

    # Start training 
    trans_error_min = np.Inf



    for epoch in range(opt.n_epochs):

        
        training_loss = 0.0
    
        model.train()
        #print("Model device:",model.device())
        print("Training epoch:", epoch)
        
        for index, (obs_batch, pose_batch, gt_trans) in enumerate(train_loader):
            print("-" * 30 + "\n")
            print(f"Now training dataset pair index: {index}")
            print(device)
            # consider adding to(device)
            obs_batch = obs_batch.to(device)
            pose_batch = pose_batch.to(device)
            gt_trans = gt_trans.to(device)

            loss = model(obs_batch, pose_batch)

            #R_pred, t_pred = model.R_est, model.t_est
            
            #identity = torch.eye(3).cuda().unsqueeze(0).repeat(opt.batch_size, 1, 1)
            #loss_supervised = F.mse_loss(torch.matmul(R_pred.transpose(2, 1), gt_trans[:, :3, :3]), identity) \
            #    + F.mse_loss(t_pred,  gt_trans[:, :3, 3].view(opt.batch_size, 3))

                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            #print('training loss', training_loss)

        training_loss_epoch = training_loss / len(train_loader)
        print('[{}/{}], training loss: {:.4f}'.format(
                epoch+1,opt.n_epochs,training_loss_epoch))

        scheduler.step()

        if (epoch+1) % opt.log_interval == 0:
            #print('[{}/{}], training loss: {:.4f}'.format(
            #    epoch+1,opt.n_epochs,training_loss_epoch))

            source_pc_est_np = []
            template_pc_np = []
            R_est_np = []
            t_est_np = []
            transformation_estimated=[]
            transformation_gt = []

            with torch.no_grad():
                model.eval()
                for index, (obs_batch_test, pose_batch_test, gt_trans) in enumerate(train_loader):
                    print("-" * 30 + "\n")
                    print(f"Now validating dataset pair index: {index}")
                    # consider adding to device
                    obs_batch_test = obs_batch_test.to(device)
                    pose_batch_test = pose_batch_test.to(device)
                    gt_trans = gt_trans.to(device)

                    model(obs_batch_test, pose_batch_test)

                    source_pc_est_np.append(model.source_pc_transform.cpu().detach().numpy())
                    template_pc_np.append(model.tmp.cpu().detach().numpy())
                    R_est_np.append(model.R_est.cpu().detach().numpy()) # shape of (4,3,3)
                    t_est_np.append(model.t_est.unsqueeze(1).cpu().detach().numpy()) # shape of (4,1,3)
                    print(f'the shape of rotation from model is{model.R_est.cpu().detach().numpy().shape}')
                    print(f'the shape of translation from model {model.t_est.unsqueeze(1).cpu().detach().numpy().shape}') # (4,1,3)
                    transformation_gt.append(gt_trans.cpu().detach().numpy())
                    trans_test=model.t_est.unsqueeze(1).cpu().detach().numpy()
                    #print(f'shape of translation test is {trans_test.shape}')
                    trans_test=np.swapaxes(trans_test,1,2)
                    print(f'shape of translation test is {trans_test.shape}')
                    stack=np.concatenate((model.R_est.cpu().detach().numpy(),trans_test),axis=-1)
                    print(f'stack shape is{stack.shape}') # shape of (1,3,4)
                    test_new=np.array([0,0,0,1]).reshape((1,1,4))
                    rep=stack.shape[0]
                    test_new=np.repeat(test_new,rep,axis=0) # shape of 4,1,4
                    #Transf=np.vstack((stack,np.array([0,0,0,1])))
                    Transf=np.concatenate((stack,test_new),axis=1)
                    print(f'the shape of transformation is {Transf.shape}')
                    transformation_estimated.append(Transf)

                R_est_np = np.concatenate(R_est_np)
                t_est_np = np.concatenate(t_est_np)
                transformation_estimated_np=np.concatenate(transformation_estimated) # from model
                print(f'the shape of transformation estimated np is {transformation_estimated_np.shape}') # (1113,4,4)
                
                transformation_gt = np.concatenate(transformation_gt)
                # add ICP code to further align the point clouds.
                
                R_icp=[]
                translation_icp=[]
                transformation_icp=[]
                print(f'the length of source pc est np is {len(source_pc_est_np)}')
                no_bat=source_pc_est_np[0][0]
                print(f'no of batches are as follows {no_bat}')
                test_transformation_icp=np.zeros((transformation_estimated_np.shape[0],4,4))
                for d in range(len(source_pc_est_np)):
                    for ba in range((source_pc_est_np[d]).shape[0]): # iterating through patches
                        print(f'shape of source pc is {source_pc_est_np[d][ba].shape}')
                        print(f'shape of target pc is {template_pc_np[d][ba].shape}')
                        rot_icp,trans_icp,transform_icp=icp_o3d(source_pc_est_np[d][ba],template_pc_np[d][ba])
                        print(f'the shape of rot_icp is {rot_icp.shape}')
                        print(f'the shape of trans_icp is {trans_icp.shape}')
                        print(f'the shape of transformation icp is {transform_icp.shape}')
                        R_icp.append(rot_icp)
                        translation_icp.append(trans_icp)
                        transformation_icp.append(transform_icp)
                        transformation_icp_np=np.concatenate(transformation_icp) # shape (number of transformations,4,4)
                        #transformation_icp_np=np.concatenate((test_transformation_icp,transformation_icp),axis=0)
                print(f'shape of transformation from icp is {transformation_icp_np.shape}')
                
                # now we need to compute the final transformation (pre-multiply transformation by transformation from icp)
                #final_transformations=np.zeros((len(transformation_icp),4,4))
                #for tr in range(len(transformation_icp)):
                    #final_transformations[tr,:,:]=transformation_icp_np[tr,:,:]@transformation_estimated_np[tr,:,:]
                
                torch_icp_transf=torch.from_numpy(transformation_icp_np)
                torch_model_transf=torch.from_numpy(transformation_estimated_np)
                final_transformations=torch.matmul(torch_icp_transf,torch_model_transf)
                final_transformations=final_transformations.numpy()
                
                R_est_new_np=final_transformations[:,:3,:3]
                t_est_new_np=final_transformations[:,:3,3]

                #avg_rot_error, avg_trans_error = eval_metric(R_est_np, t_est_np, transformation_gt)
                avg_rot_error, avg_trans_error = eval_metric(R_est_new_np, t_est_new_np, transformation_gt)


                if avg_trans_error <= trans_error_min:
                    print('Translation error decreased ({:.6f} --> {:.6f}. Saving model...'.format(trans_error_min, avg_trans_error))
                    save_name = os.path.join(checkpoint_dir,'model_best.pth')
                    utils.save_checkpoint(save_name,model,optimizer)
                    trans_error_min = avg_trans_error

                    source_pc_est_np = np.concatenate(source_pc_est_np)
                    template_pc_np = np.concatenate(template_pc_np)

                    kwargs = {'e':epoch+1}

                    utils.plot_global_point_cloud_KITTI(source_pc_est_np[1, :, :], template_pc_np[1, :, :], checkpoint_dir, plot_num=1, **kwargs)
                    utils.plot_global_point_cloud_KITTI(source_pc_est_np[10, :, :], template_pc_np[10, :, :], checkpoint_dir, plot_num=2, **kwargs)
                    utils.plot_global_point_cloud_KITTI(source_pc_est_np[30, :, :], template_pc_np[30, :, :], checkpoint_dir, plot_num=3, **kwargs)
                
#python main.py -b 9 &> results/logs/$(date +"%Y-%m-%dT%H:%M:%S").log &
