import unittest

import torch

from endpoint_contract_v1 import balanced_eos_loss, read_endpoint, endpoint_training_loss
from model import MaskedQualityGRU, Scaling
from test_training import fixture


class EndpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_ideal_target_endpoint_roundtrip(self):
        batch=fixture()
        out=read_endpoint(batch['real'])
        self.assertTrue(torch.equal(out['eos_index'],batch['length']-1))
        self.assertTrue(torch.equal(out['success_decision'],batch['real'][:,0,1].bool()))

    def test_first_crossing_wins(self):
        traj=torch.zeros(1,8,15)
        traj[0,:,0]=torch.tensor([.1,.2,.6,.1,.8,.9,.1,.7])
        traj[0,:,1]=.8
        self.assertEqual(read_endpoint(traj)['eos_index'].item(),2)

    def test_hard_readout_ignores_tail_and_extension(self):
        traj=torch.zeros(1,8,15); traj[0,2,0]=.8; traj[0,:3,1]=.75
        out=read_endpoint(traj)
        modified=torch.cat((traj,torch.zeros(1,40,15)),1)
        modified[:,3:]=float('nan')
        other=read_endpoint(modified)
        for key in out:
            self.assertTrue(torch.equal(out[key],other[key]),key)

    def test_missing_endpoint_is_explicit(self):
        traj=torch.zeros(1,8,15); traj[:,:,1]=.99
        out=read_endpoint(traj)
        self.assertFalse(out['endpoint_found'].item())
        self.assertEqual(out['eos_index'].item(),-1)
        self.assertTrue(torch.isnan(out['t_eos_s']).all())
        self.assertTrue(torch.isnan(out['success_probability']).all())
        self.assertFalse(out['success_decision'].item())

    def test_predicted_not_target_prefix_is_used(self):
        traj=torch.zeros(1,8,15); traj[0,4,0]=.7
        traj[0,:5,1]=torch.tensor([.1,.2,.3,.4,.5]); traj[0,5:,1]=1
        self.assertAlmostEqual(read_endpoint(traj)['success_probability'].item(),.3,places=6)

    def test_balanced_loss_independent_of_negative_count(self):
        values=[]
        for count in (3,30,300):
            target=torch.zeros(1,count); target[:,-1]=1
            values.append(balanced_eos_loss(torch.full_like(target,.2),target,torch.ones_like(target,dtype=torch.bool)))
        self.assertTrue(torch.allclose(torch.stack(values),values[0].expand(3)))

    def test_balanced_loss_masks_padding_gradient(self):
        target=torch.tensor([[0.,0.,1.,1.,1.]])
        probability=torch.tensor([[.2,.2,.6,float('nan'),float('nan')]],requires_grad=True)
        mask=torch.tensor([[True,True,True,False,False]])
        loss=balanced_eos_loss(probability,target,mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(probability.grad[~mask]==0))
        self.assertNotEqual(probability.grad[0,2].item(),0)

    def test_nonfinite_before_endpoint_rejected(self):
        traj=torch.zeros(1,8,15); traj[0,0,0]=float('nan');traj[0,2,0]=.8
        with self.assertRaises(ValueError): read_endpoint(traj)

    def test_no_positive_eos_is_rejected(self):
        with self.assertRaises(ValueError):
            balanced_eos_loss(torch.full((1,3),.5),torch.zeros(1,3),torch.ones(1,3,dtype=torch.bool))

    def test_new_loss_backward(self):
        batch=fixture()
        model=MaskedQualityGRU(Scaling.fit(batch),hidden=8,layers=2,dropout=0)
        pred=model(batch['inputs'],batch['point_start'],batch['sim'])
        loss,_=endpoint_training_loss(model,pred,batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
