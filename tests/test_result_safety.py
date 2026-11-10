import sys
import types
import unittest
from unittest.mock import Mock, patch

import test_engine


class ResultSafetyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_engine.Tests(methodName='runTest')
        self.fixture.setUp()
        self.engine = self.fixture.engine
        self.p = self.fixture.p

    def test_duplicate_callback_image_identity_keeps_callback_order(self):
        shared_image = types.SimpleNamespace(info={})
        result = types.SimpleNamespace(images=[shared_image, shared_image], index_of_first_image=0,
            all_seeds=[1, 2], all_prompts=['one', 'two'], infotexts=['one', 'two'],
            info='one', extra_generation_params={})
        self.p._pi_deferred_slots = [types.SimpleNamespace(image=shared_image, index=0),
                                     types.SimpleNamespace(image=shared_image, index=1)]
        seen = []
        script = types.SimpleNamespace(postprocess_image_after_composite=lambda p, pp: seen.append(pp.index))
        with patch.object(self.engine, 'Script', return_value=script):
            self.engine._finish_images(self.p, result)
        self.assertEqual(seen, [0, 1])

    def test_grid_save_failure_keeps_original_grid_and_reports_partial_success(self):
        base_grid = types.SimpleNamespace(info={'parameters':'base grid'})
        sources = [types.SimpleNamespace(info={}), types.SimpleNamespace(info={})]
        result = types.SimpleNamespace(images=[base_grid, *sources], index_of_first_image=1,
            all_seeds=[1, 2], all_prompts=['one', 'two'],
            infotexts=['base grid', 'one', 'two'], info='base grid', comments='',
            extra_generation_params={})
        self.p._pi_saved_files = {id(base_grid):[('grid.png', None)]}
        self.p._pi_saved_grids = [(id(base_grid), 'base grid')]
        replacements = []

        def refine(p, pp):
            image = types.SimpleNamespace(info={})
            replacements.append(image)
            pp.image = image
            p.extra_generation_params['Refiner'] = 'target'

        def rewrite(image, info, records):
            if records:
                raise OSError('grid disk full')

        grid = types.SimpleNamespace(info={})
        self.fixture.modules.images = types.SimpleNamespace(image_grid=Mock(return_value=grid))
        script = types.SimpleNamespace(postprocess_image_after_composite=refine)
        with patch.object(self.engine, 'Script', return_value=script), \
             patch.object(self.engine, '_rewrite_saved', side_effect=rewrite):
            returned = self.engine._finish_images(self.p, result)

        self.assertIs(returned.images[0], base_grid)
        self.assertEqual(returned.infotexts[0], 'base grid')
        self.assertEqual(returned.info, 'base grid')
        self.assertEqual(returned.images[1:], replacements)
        self.assertIn('Grid: saving refined grid failed: grid disk full',
                      returned.extra_generation_params['Refiner error'])


if __name__ == '__main__':
    unittest.main()
