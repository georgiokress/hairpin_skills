import unittest

import torch

from soft_endpoint_v1 import soft_endpoint,objective_terms


class SoftEndpointTests(unittest.TestCase):
    def test_ideal_step_has_exact_endpoint_and_prefix_success(self):
        x=torch.zeros(1,20,15,dtype=torch.float64);x[:,5:,0]=1;x[:,:6,1]=.8
        result=soft_endpoint(x,threshold=.95)
        self.assertAlmostEqual(result['duration_s'].item(),5*.032,places=10)
        self.assertAlmostEqual(result['success_score'].item(),.8,places=10)
        self.assertEqual(result['missing_endpoint_mass'].item(),0.)

    def test_no_end_has_horizon_time_and_failure(self):
        x=torch.zeros(1,640,15);x[:,:,1]=1
        result=soft_endpoint(x)
        self.assertAlmostEqual(result['duration_s'].item(),639*.032,places=5)
        self.assertEqual(result['success_score'].item(),0)
        self.assertEqual(result['missing_endpoint_mass'].item(),1)

    def test_constant_subthreshold_score_does_not_accumulate(self):
        values=[]
        for length in (10,640):
            x=torch.zeros(1,length,15);x[:,:,0]=.9
            values.append(soft_endpoint(x,threshold=.95)['endpoint_mass'].item())
        self.assertAlmostEqual(*values,places=7)
        self.assertLess(values[0],.01)

    def test_decreasing_eos_does_not_restart_clock(self):
        x=torch.zeros(1,20,15,dtype=torch.float64);x[:,5,0]=1;x[:,:6,1]=.8
        result=soft_endpoint(x)
        self.assertAlmostEqual(result['duration_s'].item(),.16,places=10)

    def test_mass_conservation(self):
        torch.manual_seed(1);x=torch.rand(4,100,15)
        result=soft_endpoint(x,threshold=.9)
        self.assertTrue(torch.all(result['weights']>=0))
        self.assertTrue(torch.allclose(result['weights'].sum(1)+result['missing_endpoint_mass'],torch.ones(4)))

    def test_gradient_matches_finite_difference(self):
        u=torch.tensor(.02,dtype=torch.float64,requires_grad=True)
        times=torch.arange(100,dtype=torch.float64)*.032
        def evaluate(value):
            x=torch.zeros(1,100,15,dtype=torch.float64)
            x[0,:,0]=torch.sigmoid(3*(times-1.2-value));x[:,:,1]=.8
            return soft_endpoint(x,threshold=.9)['duration_s'].sum()
        actual=torch.autograd.grad(evaluate(u),u)[0]
        step=1e-6;expected=(evaluate(u.detach()+step)-evaluate(u.detach()-step))/(2*step)
        self.assertTrue(torch.isfinite(actual))
        self.assertGreater(abs(actual.item()),1e-4)
        self.assertTrue(torch.allclose(actual,expected,rtol=1e-4,atol=1e-5))

    def test_invalid_values_rejected(self):
        x=torch.zeros(1,10,15);x[:,0,0]=float('nan')
        with self.assertRaises(ValueError):soft_endpoint(x)

    def test_terms_do_not_invent_preference_weights(self):
        x=torch.zeros(1,10,15);x[:,5:,0]=1;x[:,:,1]=.8
        result=objective_terms(dict(trajectory=x,error_mm=torch.tensor([.3])))
        self.assertNotIn('total',result)
        self.assertAlmostEqual(result['absolute_error_normalized'].item(),.2,places=6)


if __name__=='__main__':unittest.main()
