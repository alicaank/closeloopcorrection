"""Source/policy attribution and immutable-output reuse checks, without a GPU."""
import json
import tempfile
import unittest
from pathlib import Path
from evidence import eligibility,receipt_paths,sha,verify_binding


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)/'run'
        self.d.mkdir()
        self.case = dict(model='matrixgame',scene='0238de14',seed=107)
        self.expected = dict(matrixgame='main-hash',matrix_child='child-hash')
        (self.d/'run.json').write_text(json.dumps(dict(**self.case,arm='baseline')))
        for model,p in receipt_paths('matrixgame',self.d).items():
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(dict(model=model,memory_policy='commanded',completed=True,overlay_sha256=self.expected[model])))

    def test_complete_matching_metadata_is_eligible_for_raw_validation(self):
        self.assertEqual(eligibility(self.case,'B',self.d,self.expected),(True,'metadata_and_parent_source_match'))

    def test_commanded_output_is_not_a_measured_arm(self):
        self.assertEqual(eligibility(self.case,'M',self.d,self.expected),(False,'reference_policy_or_completion_mismatch'))

    def test_child_driver_hash_is_required(self):
        expected = dict(self.expected,matrix_child='another-version')
        self.assertEqual(eligibility(self.case,'B',self.d,expected),(False,'reference_generated_source_mismatch'))

    def test_different_seed_is_not_reused(self):
        self.assertEqual(eligibility(dict(self.case,seed=7),'B',self.d,self.expected),(False,'reference_run_identity_mismatch'))

    def test_partial_child_and_missing_child_are_not_reused(self):
        p = receipt_paths('matrixgame',self.d)['matrix_child']
        d = json.loads(p.read_text());d['completed']=False;p.write_text(json.dumps(d))
        self.assertFalse(eligibility(self.case,'B',self.d,self.expected)[0])
        p.unlink()
        self.assertEqual(eligibility(self.case,'B',self.d,self.expected),(False,'reference_missing_or_incomplete'))

    def test_lingbot_receipts_live_above_native_arm_directory(self):
        d = self.d/'baseline'
        self.assertEqual(receipt_paths('lingbot',d)['lingbot'],self.d/'memory_policy_lingbot.json')

    def test_lyra_requires_both_parent_and_child_receipts(self):
        self.assertEqual(set(receipt_paths('lyra',self.d)),{'lyra','lyra_child'})

    def test_binding_rejects_modified_artifacts(self):
        p = self.d/'run.json'
        binding = dict(files={str(p):sha(p)})
        verify_binding(binding)
        p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Bound output changed'):
            verify_binding(binding)


if __name__=='__main__': unittest.main(verbosity=2)
