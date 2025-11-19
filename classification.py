import torch
import torch.nn.functional as F
import ocnn

from thsolver import Solver
from datasets import get_modelnet40_dataset,get_tetmesh_dataset
from builder import get_classification_model


class ClsSolver(Solver):
  
  def get_model(self, flags):
    return get_classification_model(flags)

  def get_dataset(self, flags):
    return get_tetmesh_dataset(flags)

  def get_input_feature(self, octree):
    flags = self.FLAGS.MODEL
    # print(flags)
    octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
    data = octree_feature(octree)
    return data

  def forward(self, batch):
    # 
    # print( batch['ptau'],  batch['label'])
    octree, octree2, label, ptau = batch['octree'].cuda(), batch['octree2'].cuda(), batch['label'].cuda(), torch.stack(batch['ptau']).cuda()
   
    # xyz = octree['xyz']  # Assuming the point cloud data is stored in 'xyz'
    
    # # Print the xyz coordinates for the current batch
    # print(f"XYZ Coordinates: {xyz}")
    data = self.get_input_feature(octree)
    data2 = self.get_input_feature(octree2)
    #print(f"data shape: {data.shape}, data2 shape: {data2.shape}, label shape: {label.shape}, ptau shape: {ptau.shape}")
    logits = self.model(data, data2, octree, octree2, octree.depth, octree2.depth, ptau)
    log_softmax = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(log_softmax, label)
    pred = torch.argmax(logits, dim=1)
    accu = pred.eq(label).float().mean()
    # Confusion matrix elements
    TP = ((pred == 1) & (label == 1)).sum().float()
    TN = ((pred == 0) & (label == 0)).sum().float()
    FP = ((pred == 1) & (label == 0)).sum().float()
    FN = ((pred == 0) & (label == 1)).sum().float()

    # Avoid divide-by-zero
    eps = 1e-6
    sen = TP / (TP + FN + eps)
    spe = TN / (TN + FP + eps)

    return loss, accu, sen, spe

  def train_step(self, batch):
    loss, accu, _, _ = self.forward(batch)
    return {'train/loss': loss, 'train/accu': accu}

  def test_step(self, batch):
    with torch.no_grad():
      loss, accu, sen, spe = self.forward(batch)
    return {'test/loss': loss, 'test/accu': accu, 'test/sen': sen, 'test/spe': spe}


if __name__ == "__main__":
  ClsSolver.main()
