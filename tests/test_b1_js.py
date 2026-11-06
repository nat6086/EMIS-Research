import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from emis_research.mismatch.b1_js import (LABELS, canonical_probabilities,
    assemble_scores, calibration_plan, fit_calibrated, js_divergence, load_b0,
    pair_with_b0, ranking_comparison, validate_probabilities)
from emis_research.mismatch.b0_labels import ROOT, load_audit


class JSTests(unittest.TestCase):
    def test_identical_disjoint_zeros_and_repeatability(self):
        p = np.eye(4)
        np.testing.assert_array_equal(js_divergence(p, p), np.zeros(4))
        np.testing.assert_array_equal(js_divergence(p, np.roll(p, 1, axis=1)), np.ones(4))
        np.testing.assert_array_equal(js_divergence(p, p), js_divergence(p, p))

    def test_symmetry_range_and_known_value(self):
        rng = np.random.default_rng(42)
        p, q = rng.dirichlet(np.ones(4), 100), rng.dirichlet(np.ones(4), 100)
        np.testing.assert_allclose(js_divergence(p, q), js_divergence(q, p), rtol=0, atol=1e-15)
        self.assertTrue(((js_divergence(p, q) >= 0) & (js_divergence(p, q) <= 1)).all())
        self.assertAlmostEqual(js_divergence([[1,0,0,0]], [[.5,.5,0,0]])[0], .31127812445913283)

    def test_invalid_probabilities(self):
        for p in [[[1, 0, 0]], [[np.nan,0,0,1]], [[np.inf,0,0,0]],
                  [[-.1,.1,0,1]], [[1.1,0,0,0]], [[.2,.2,.2,.2]]]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                validate_probabilities(p)

    def test_class_order(self):
        classes = LABELS[::-1]
        p = canonical_probabilities([[.1,.2,.3,.4]], classes)
        np.testing.assert_allclose(p, [[.4,.3,.2,.1]])
        with self.assertRaises(ValueError):
            canonical_probabilities([[.1,.2,.3,.4]], ['sad'] * 4)

    def test_real_b0_coverage_and_pairing(self):
        d, _, q, _ = load_audit()
        b = pd.read_csv(ROOT / 'outputs/mismatch_v1/mismatch_scores.csv')
        scored = d[['audio_id']].assign(protocol='S1', outer_fold=d.fold_S1, role='test',
            y_mismatch=(d.text_emotion != d.generation_emotion).astype(int), b1_score=.5)
        paired = pair_with_b0(scored, b.iloc[::-1])
        self.assertEqual(len(paired), 1248)
        self.assertTrue(paired.audio_id.is_unique)
        self.assertEqual(len(q), 104)
        comparison = ranking_comparison(paired)
        self.assertEqual(len(comparison), 12)
        for bad in [b.iloc[:-1], pd.concat([b,b.iloc[:1]]), b.assign(y_mismatch=1),
                    b.assign(outer_fold=(b.outer_fold+1)%5)]:
            with self.assertRaises(ValueError):
                pair_with_b0(scored, bad)

    def test_existing_selections_and_all_calibration_partitions(self):
        d, _, _, audit = load_audit()
        baseline, selections, _ = load_b0(d, audit)
        self.assertEqual(len(baseline), 1248)
        for r in selections:
            self.assertEqual(r['layer'], 3)
            tr, te, splits = calibration_plan(d, r['outer_fold'])
            self.assertEqual(len(splits), 5)
            for a,b in splits:
                self.assertFalse(set(tr[a]) & set(te))
                self.assertFalse(set(tr[b]) & set(te))
                self.assertFalse(set(d.iloc[tr[a]].group_text) & set(d.iloc[tr[b]].group_text))

    def test_synthetic_calibration_no_test_influence_both_schemes(self):
        rng = np.random.default_rng(42)
        d = pd.DataFrame({'audio_id': [f'a{i}' for i in range(64)],
             'group_text': np.repeat([f'g{i}' for i in range(16)], 4),
             'generation_emotion': LABELS * 16, 'fold_S1': np.repeat([0,1], 32)})
        cube = rng.normal(size=(64, 1, 3))
        selection = dict(outer_fold=0, layer=0, params={'C':1, 'gamma':.1})
        changed = d.copy(); changed.loc[changed.fold_S1 == 0, 'generation_emotion'] = 'sad'
        cube2 = cube.copy(); cube2[d.fold_S1 == 0] = 10000
        for ensemble in [False, True]:
            rows, record, model = fit_calibrated(d, cube, selection, ensemble)
            _, record2, model2 = fit_calibrated(changed, cube2, selection, ensemble)
            self.assertEqual(record, record2)
            self.assertEqual(len(rows), 32)
            for a,b in zip(model.calibrated_classifiers_, model2.calibrated_classifiers_):
                np.testing.assert_array_equal(a.estimator['scale'].mean_, b.estimator['scale'].mean_)
                np.testing.assert_array_equal(a.estimator['clf'].support_vectors_, b.estimator['clf'].support_vectors_)
                np.testing.assert_array_equal(a.estimator['clf'].dual_coef_, b.estimator['clf'].dual_coef_)
                for ca,cb in zip(a.calibrators,b.calibrators):
                    self.assertEqual(ca.a_, cb.a_)
                    self.assertEqual(ca.b_, cb.b_)
        with self.assertRaises(ValueError):
            fit_calibrated(d,cube,selection,None)

    def test_full_join_with_artificial_audio_probabilities(self):
        # No són prediccions experimentals: comprovació del contracte de sortida.
        d, _, text, audit = load_audit()
        baseline, _, _ = load_b0(d,audit)
        from emis_research.mismatch.b1_js import PA
        audio = d[['audio_id']].assign(protocol='S1',outer_fold=d.fold_S1,role='test',
            pred_audio_calibrated='angry', **{c:.25 for c in PA})
        result = assemble_scores(d,audio,text,baseline)
        self.assertEqual(len(result),1248)
        self.assertTrue(result.b1_score.between(0,1).all())
        with self.assertRaises(ValueError):
            assemble_scores(d,audio.iloc[:-1],text,baseline)


if __name__ == '__main__':
    unittest.main()
