"""Robot-free regression and interchange tests with synthetic data."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tables

import analyze_export as export


class OfflineExportTests(unittest.TestCase):
    def test_combination_grouping(self):
        rows = [dict(run_id=f'F{f}v{v}b{b}', F_N=f, v_nominal_mm_s=v)
                for b in (1,2,3) for f in (1,1.5,2,3,4,5) for v in (1,2,4,6,10)]
        folds = export.grouped_folds(rows)
        self.assertEqual(folds, export.grouped_folds(rows))
        all_held = []
        for fold in folds:
            self.assertEqual(len(fold['train_indices']),72)
            self.assertEqual(len(fold['heldout_indices']),18)
            tr = {(rows[i]['F_N'],rows[i]['v_nominal_mm_s']) for i in fold['train_indices']}
            ho = {(rows[i]['F_N'],rows[i]['v_nominal_mm_s']) for i in fold['heldout_indices']}
            self.assertFalse(tr & ho)
            all_held.extend(fold['heldout_indices'])
        self.assertEqual(sorted(all_held),list(range(90)))

    def fixture(self, root, trigger):
        folder = root / ('trigger' if trigger else 'no_trigger')
        folder.mkdir()
        t = np.arange(0,15.001,.002)
        d = .001 * t
        start = .0002
        pos = np.column_stack((start+d,np.zeros_like(t),-.015*d))
        if trigger:
            pos[:,2] -= np.maximum(d-.00925,0) * .4
        meta = dict(run_id=folder.name,purpose='training',technical_valid=True,
                    v_slide_mm_s=1,f_n_mag_N=2,r_gt_run_mm=9.05,r_gt_nom_mm=9.25,
                    point_start=[start,0,0,1,0,0,0],x_r_start_m=start,operator_slip=False,
                    logical_run_id=folder.name,attempt_index=1,sequence_index=4,
                    combination_id='D11',block=1,order_in_block=4,actual_start_utc=None,
                    hairpin_id=0,setup_id=None,prior_reuse_count=None,force_command_N=-2,
                    slip_basis='assumed',slip_observation_confirmed=False,slip_source='USER_POLICY_ASSUMPTION',
                    contact_status='unqualified',qc=dict(force_tracking_warning=False),
                    source=dict(manifest_sha256='fixture',collector_bag_sha256='fixture',operator_annotation_sha256='fixture'))
        (folder/'meta.json').write_text(json.dumps(meta))
        np.savez_compressed(folder/'series.npz',t=t,pos=pos,quat=np.tile([1,0,0,0],(len(t),1)),
                            wrench=np.tile([0,0,2,0,0,0],(len(t),1)))
        return folder

    def test_native_label_EOS_padding_and_loader(self):
        with tempfile.TemporaryDirectory(prefix='main90_export_test_') as temp:
            root=Path(temp)
            for trigger in (True,False):
                folder=self.fixture(root,trigger)
                entry,reason=export.convert.convert_run(str(folder),export.PilotDetectorConfig(),export.SequenceConfig())
                self.assertIsNone(reason)
                row=export.labels_and_stats(entry,folder)
                self.assertEqual(row['triggered'],trigger)
                self.assertEqual(row['r_GT_run_mm'],9.05)
                if trigger:
                    self.assertGreaterEqual(row['eos_grid_s'],row['t_det_native_s'])
                    self.assertLess(row['eos_grid_s']-row['t_det_native_s'],.032+1e-8)
                    self.assertAlmostEqual(row['e_signed_mm'],row['r_det_relative_mm']-9.05)
                else:
                    self.assertIsNone(row['e_abs_mm'])
                    self.assertEqual(row['success'],0)
                    self.assertLessEqual(row['eos_grid_s'],15)
                self.assertFalse(row['slip_observation_confirmed'])
                out=export.save_h5([entry],root/f'{trigger}.h5')
                self.assertTrue(out['loader_verified'])
                with tables.open_file(out['path']) as f:
                    self.assertEqual(int(f.get_node('/valid_length/0').read()[0]),row['valid_length'])
                    self.assertEqual(int(f.get_node('/physical_sample_mask/0').read().sum()),row['valid_length'])
                    self.assertFalse(bool(f.root.run_metadata.slip_observation_confirmed.read()[0]))


if __name__=='__main__':
    unittest.main()
