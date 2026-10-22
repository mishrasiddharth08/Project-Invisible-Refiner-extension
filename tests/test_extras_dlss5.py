import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image


ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExtrasDlss5Tests(unittest.TestCase):
    def setUp(self):
        modules = types.ModuleType('modules')
        self.errors = types.SimpleNamespace(report=Mock())
        modules.errors = self.errors
        modules.scripts = types.SimpleNamespace(scripts_data=[])
        modules.scripts_postprocessing = types.SimpleNamespace(ScriptPostprocessing=object)
        self.state = types.SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False)
        modules.shared = types.SimpleNamespace(state=self.state)
        self.memory = types.SimpleNamespace(unload_all_models=Mock(), soft_empty_cache=Mock())
        self.patch = patch.dict(sys.modules, {
            'modules': modules,
            'backend': types.SimpleNamespace(memory_management=self.memory),
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.entry = load('extras_dlss5_test', ROOT / 'scripts' / 'postprocessing_dlss5.py')
        self.original = Image.new('RGBA', (4, 3), (10, 20, 30, 77))
        self.pp = types.SimpleNamespace(image=self.original, info={})

    def test_disabled_is_noop(self):
        self.entry._backend = Mock(side_effect=AssertionError('backend loaded'))
        self.entry.ScriptPostprocessingDLSS5().process(self.pp, False, 1)
        self.assertIs(self.pp.image, self.original)

    def test_success_publishes_image_and_metadata(self):
        replacement = Image.new('RGBA', self.original.size, (40, 50, 60, 77))
        backend = types.SimpleNamespace(TARGET='DLSS 5', enhance_image=Mock(return_value=replacement))
        self.entry._backend = Mock(return_value=backend)
        self.entry._release_gpu = Mock()
        self.entry._native_video_call = Mock(return_value=False)
        self.entry.ScriptPostprocessingDLSS5().process(self.pp, True, .75)
        self.assertIs(self.pp.image, replacement)
        self.assertEqual(self.pp.info['DLSS 5'], 'intensity 0.75')
        self.entry._release_gpu.assert_called_once_with('DLSS 5')
        self.assertTrue(callable(backend.enhance_image.call_args.kwargs['cancelled']))

    def test_failure_retains_original(self):
        backend = types.SimpleNamespace(TARGET='DLSS 5', enhance_image=Mock(side_effect=RuntimeError('failed')))
        self.entry._backend = Mock(return_value=backend)
        self.entry._release_gpu = Mock()
        self.entry._native_video_call = Mock(return_value=False)
        warning = Mock()
        with patch.object(self.entry.gr, 'Warning', warning):
            self.entry.ScriptPostprocessingDLSS5().process(self.pp, True, 1)
        self.assertIs(self.pp.image, self.original)
        self.errors.report.assert_called_once()
        warning.assert_called_once()

    def test_native_video_is_skipped_without_loading_backend(self):
        self.entry._native_video_call = Mock(return_value=True)
        self.entry._backend = Mock(side_effect=AssertionError('backend loaded'))
        script = self.entry.ScriptPostprocessingDLSS5()
        script.process(self.pp, True, 1)
        script.process(self.pp, True, 1)
        self.assertIs(self.pp.image, self.original)
        self.errors.report.assert_called_once()


if __name__ == '__main__':
    unittest.main()
