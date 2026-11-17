import unittest
import torch

from calibrate_endpoint_v1 import selection_score
from invert_candidate_pareto_v1 import nondominated


class SelectionTests(unittest.TestCase):
    def test_missing_endpoint_selection_penalty(self):
        result=selection_score(dict(endpoint_found=torch.tensor([True,False]),
            t_eos_s=torch.tensor([1.1,float('nan')])),torch.tensor([1.,2.]))
        self.assertAlmostEqual(result,(.1+20.448)/2,places=5)

    def test_pareto_keeps_tradeoff_not_dominated_point(self):
        a=dict(error_mm=.1,time_s=4.);b=dict(error_mm=.5,time_s=1.);c=dict(error_mm=.7,time_s=5.)
        self.assertEqual(nondominated([a,b,c]),[a,b])

    def test_empty_pareto(self):self.assertEqual(nondominated([]),[])


if __name__=='__main__':unittest.main()
