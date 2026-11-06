"""Tests sintètics petits i auditoria local; cap entrenament del corpus."""
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from emis_research.mismatch.b0_labels import (LABELS, align_inputs, fit_outer,
    dump_new, load_audit, metrics_by_fold, score_inputs)


class B0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d, cls.cube, cls.text, cls.audit = load_audit()

    def acoustic(self):
        return self.d[['audio_id']].assign(protocol='S1', outer_fold=self.d.fold_S1,
                   role='test', pred_audio=self.d.generation_emotion)

    def test_real_coverage_alignment_and_splits(self):
        self.assertEqual(len(self.d), 1248)
        self.assertEqual(self.d.group_text.nunique(), 104)
        for k in range(5):
            a, b = self.d[self.d.fold_S1 != k], self.d[self.d.fold_S1 == k]
            self.assertFalse(set(a.group_text) & set(b.group_text))
            self.assertFalse(set(a.audio_id) & set(b.audio_id))
        self.assertEqual(self.audit['y_counts'], {'1': 936, '0': 312})

    def test_b0_equal_different_and_ground_truth_independent(self):
        a = self.acoustic()
        expected = self.d.text.map(self.text.set_index('text').pred_text)
        a['pred_audio'] = expected.to_numpy()
        scored = score_inputs(self.d, a, self.text)
        self.assertTrue(scored.b0_score.eq(0).all())
        a['pred_audio'] = [LABELS[(LABELS.index(x) + 1) % 4] for x in a.pred_audio]
        changed = score_inputs(self.d, a, self.text)
        self.assertTrue(changed.b0_score.eq(1).all())
        np.testing.assert_array_equal(scored.y_mismatch, changed.y_mismatch)

    def test_missing_duplicate_wrong_fold_and_wrong_role_rejected(self):
        a = self.acoustic()
        variants = [a.iloc[:-1], pd.concat([a, a.iloc[:1]]),
                    a.assign(outer_fold=(a.outer_fold + 1) % 5), a.assign(role='train_oof')]
        for bad in variants:
            with self.subTest(kind=len(bad)):
                with self.assertRaises(ValueError):
                    score_inputs(self.d, bad, self.text)
        with self.assertRaises(ValueError):
            score_inputs(self.d, a, self.text.iloc[:-1])

    def test_join_order_invariance(self):
        a = self.acoustic()
        first = score_inputs(self.d, a, self.text).set_index('audio_id').sort_index()
        second = score_inputs(self.d, a.iloc[::-1], self.text.iloc[::-1]).set_index('audio_id').sort_index()
        pd.testing.assert_frame_equal(first, second)

    def test_input_misalignment_rejected_and_feature_order_resolved(self):
        root = Path(__file__).resolve().parents[1]
        m = pd.read_csv(root / 'data/processed/emis_manifest.csv', dtype=str)
        s = pd.read_csv(root / 'configs/splits/splits_v1.csv', dtype={'audio_id': str, 'reference_id': str})
        ids = m.audio_id.to_numpy()
        d, _, _ = align_inputs(m, s, ids[::-1], self.text)
        np.testing.assert_array_equal(ids[::-1][d.feature_row], d.audio_id)
        bad = s.copy(); bad.loc[0, 'group_text'] = 'wrong'
        with self.assertRaises(ValueError):
            align_inputs(m, bad, ids, self.text)
        bad = s.copy(); bad.loc[0, 'fold_S1'] = (bad.loc[0, 'fold_S1'] + 1) % 5
        with self.assertRaises((ValueError, AssertionError)):
            align_inputs(m, bad, ids, self.text)
        with self.assertRaises(ValueError):
            align_inputs(m, s, np.append(ids[:-1], ids[0]), self.text)

    def test_binary_metrics_known_case_and_single_class(self):
        s = pd.DataFrame({'outer_fold': [0]*4, 'y_mismatch': [0, 0, 1, 1], 'b0_score': [0, 1, 0, 1]})
        r = metrics_by_fold(s).iloc[0]
        self.assertEqual(r.auroc, .5)
        self.assertEqual(r.average_precision, .5)
        self.assertEqual(r.balanced_accuracy, .5)
        self.assertEqual(r.f1, .5)
        r = metrics_by_fold(s.assign(y_mismatch=1)).iloc[0]
        self.assertTrue(pd.isna(r.auroc))
        self.assertIn('single', r.status)

    def test_text_tie_policy_and_invalid_probabilities(self):
        root = Path(__file__).resolve().parents[1]
        m = pd.read_csv(root / 'data/processed/emis_manifest.csv', dtype=str)
        s = pd.read_csv(root / 'configs/splits/splits_v1.csv', dtype={'audio_id': str, 'reference_id': str})
        q = self.text.copy()
        q.loc[0, [f'p_{c}' for c in LABELS]] = .25
        _, text, ties = align_inputs(m, s, m.audio_id.to_numpy(), q)
        self.assertEqual(text.iloc[0].pred_text, 'angry')
        self.assertEqual(ties, 1)
        q.loc[0, 'p_angry'] = np.nan
        with self.assertRaises(ValueError):
            align_inputs(m, s, m.audio_id.to_numpy(), q)

    def test_artifacts_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'audit.json'
            dump_new(p, {'original': True})
            original = p.read_bytes()
            with self.assertRaises(FileExistsError):
                dump_new(p, {'original': False})
            self.assertEqual(p.read_bytes(), original)

    def test_test_mutation_cannot_change_learned_parameters(self):
        # 12 petits grups, cadascun amb quatre classes; dos folds externs.
        rng = np.random.default_rng(42)
        n = 48
        d = pd.DataFrame({'audio_id': [f'a{i}' for i in range(n)],
            'group_text': np.repeat([f'g{i}' for i in range(12)], 4),
            'generation_emotion': LABELS * 12, 'fold_S1': np.repeat([0, 1], 24)})
        x = rng.normal(size=(n, 1, 3))
        configs = [('SVM-RBF', {'C': c, 'gamma': .1}, 0) for c in [1, 10]]
        _, record, model = fit_outer(d, x, 0, configs)
        changed = d.copy(); changed.loc[changed.fold_S1 == 0, 'generation_emotion'] = 'sad'
        x2 = x.copy(); x2[d.fold_S1 == 0] = 10000
        _, record2, model2 = fit_outer(changed, x2, 0, configs)
        self.assertEqual(record, record2)
        np.testing.assert_array_equal(model['scale'].mean_, model2['scale'].mean_)
        np.testing.assert_array_equal(model['scale'].scale_, model2['scale'].scale_)
        np.testing.assert_array_equal(model['clf'].support_vectors_, model2['clf'].support_vectors_)
        np.testing.assert_array_equal(model['clf'].dual_coef_, model2['clf'].dual_coef_)
        np.testing.assert_array_equal(model['clf'].intercept_, model2['clf'].intercept_)
        test_ids = set(record['test_audio_ids'])
        for part in record['inner_splits']:
            self.assertFalse(test_ids & set(part['train_audio_ids']))
            self.assertFalse(test_ids & set(part['validation_audio_ids']))


if __name__ == '__main__':
    unittest.main()
