import unittest
import json
from pathlib import Path
import torch

from analytical_prior_v1 import sliding_prior
from compare_quality_cv_v2 import BASE, MAIN, validate_sources
from model import MaskedQualityGRU,Scaling
from soft_endpoint_v1 import objective_terms


@unittest.skipUnless(
    (MAIN/'offline_export_summary_v1.json').is_file(),
    'frozen Development dataset is not available',
)
class PriorTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_matches_frozen_development_prior(self):
        _,data,_=validate_sources()
        base=BASE
        reference=[]
        for rid in data['run_ids']:
            folder=base/('main_doe_v1_spi_v1/native_runs' if '_maindoev1_' in rid else 'pilot_doe_v0_spi_v1/exported_runs')/rid
            meta=json.loads((folder/'meta.json').read_text())
            reference.append(meta['r_gt_run_mm']*.001)
        actual=sliding_prior(data['inputs'],data['point_start'],reference)
        self.assertTrue(torch.allclose(actual,data['sim'],rtol=1e-5,atol=1e-6),
            f"Maximum deviation: {(actual-data['sim']).abs().max()}")

    def test_actual_parameter_gradients_through_prior_and_gru(self):
        main,_,_=validate_sources()
        torch.manual_seed(123)
        model=MaskedQualityGRU(Scaling.fit(main),hidden=8,layers=2,dropout=0).double().eval()
        for p in model.parameters():p.requires_grad_(False)
        x=torch.tensor([[.004,2.]],dtype=torch.float64,requires_grad=True)
        start=main['point_start'][:1].double()
        def loss(x):
            pred=model(x,start,sliding_prior(x,start,.00925))
            terms=objective_terms(pred)
            return (terms['failure']+.3*terms['duration_normalized']+terms['absolute_error_normalized']).sum()
        gradient=torch.autograd.grad(loss(x),x)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        for j,delta in enumerate((1e-7,1e-5)):
            plus=x.detach().clone();minus=x.detach().clone();plus[0,j]+=delta;minus[0,j]-=delta
            fd=(loss(plus)-loss(minus))/(2*delta)
            self.assertTrue(torch.allclose(gradient[0,j],fd,rtol=3e-3,atol=1e-4))


if __name__=='__main__':unittest.main()
