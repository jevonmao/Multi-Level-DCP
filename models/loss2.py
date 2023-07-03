import torch
import torch.nn as nn

torch.manual_seed(69)
INF = 1000000

class BCEWithLogitsLoss2(nn.Module):
    def __init__(self, weight=None, reduction='elementwise_mean'):
        super(BCEWithLogitsLoss2, self).__init__()
        self.reduction = reduction
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return bce_with_logits(input, target, weight=self.weight, reduction=self.reduction)


def bce_with_logits(input, target, weight=None, reduction='elementwise_mean'):
    """
    This function differs from F.binary_cross_entropy_with_logits in the way 
    that if weight is not None, the loss is normalized by weight
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))
    if weight is not None:
        if not (weight.size() == input.size()):
            raise ValueError("Weight size ({}) must be the same as input size ({})".format(
                weight.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + \
        ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'elementwise_mean':
        if weight is not None:
            # different from F.binary_cross_entropy_with_logits
            return loss.sum() / weight.sum()
        else:
            return loss.mean()
    else:
        return loss.sum()


def bce(pred, targets, weight=None):
    criternion = BCEWithLogitsLoss2(weight=weight)
    loss = criternion(pred, targets)
    return loss

class ChamferLoss(nn.Module):
    
	def __init__(self):
		super(ChamferLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()

	def forward(self,preds,gts):
		P = self.batch_pairwise_dist(gts, preds)
		mins, _ = torch.min(P, 1)#[b,n]
		loss_1 = torch.sum(mins)
		mins, _ = torch.min(P, 2)
		loss_2 = torch.sum(mins)

		return loss_1 + loss_2


	def batch_pairwise_dist(self,x,y):
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2,1)) # generates a square matrix of size (batches,no of points,no of points)
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1)) # of size (batches,no of points,no of points)
		if self.use_cuda:
			dtype = torch.cuda.LongTensor
		else:
			dtype = torch.LongTensor
		diag_ind_x = torch.arange(0, num_points_x).type(dtype) # 1d tensor of all indices.
		diag_ind_y = torch.arange(0, num_points_y).type(dtype)
		#brk()
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P

class GlobalAlignLoss(nn.Module):
    
	def __init__(self):
		super(GlobalAlignLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

	def forward(self,preds,gts, c):
		P = self.batch_pairwise_dist(gts, preds)
		mins, _ = torch.min(P, 1)
		mins = self.huber_loss(mins,c)
		loss_1 = torch.sum(mins)
		mins, _ = torch.min(P, 2)
		mins = self.huber_loss(mins,c)
		loss_2 = torch.sum(mins)

		return loss_1 + loss_2

	def huber_loss(self,x,c):
		x = torch.where(x<c,0.5*(x**2),c*x-0.5*(c**2))
		return x

	def batch_pairwise_dist(self,x,y):
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		if self.use_cuda:
			dtype = torch.cuda.LongTensor
		else:
			dtype = torch.LongTensor
		diag_ind_x = torch.arange(0, num_points_x, device=self.device).type(dtype)
		diag_ind_y = torch.arange(0, num_points_y, device=self.device).type(dtype)
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P


def bce_ch(src_trans, tgt, src_key_trans, tgt_key, src_key_knn, tgt_key_knn, pred, targets, alpha=0.90):

	GAL = GlobalAlignLoss()
	#ch = ChamferLoss()
	criterion = nn.MSELoss(reduction='none')
	bce_loss = bce(pred, targets)
	gal_loss = GAL(src_trans, tgt, 0.01)


	keypoints_loss = criterion(src_key_trans, tgt_key).sum(1).sum(1).mean()
	knn_consensus_loss = criterion(src_key_knn, tgt_key_knn).sum(1).sum(1).mean()
	neighborhood_consensus_loss = knn_consensus_loss/8 + keypoints_loss


	#chamfer_loss = ch(source, template)

	print('Gal loss:', gal_loss) # investigate gal loss and chamfer loss similarities. 
	print('bce loss:', bce_loss) # works best for dense frames with good initialization. 
	print('neighborhood loss:', neighborhood_consensus_loss)


	return gal_loss 

'''
want to implement supervised loss function that computes the difference between 
the ground truth poses and predicted poses in a similar manner to DCP. 
I should focus on this for this time. 
'''

class SupervisedLoss(nn.Module):
	def __init__(self):
		super(SupervisedLoss, self).__init__()
		self.use_cuda = torch.cuda.is_available()
	
	def forward(self,preds,gts):
		pass
		'''
		predictions are the output from the network. should be
		a transformation matrix of shape (4 by 4 by number of examples or sequences)
		and gts are the ground truth pose estimations given of the same shape.
		'''


