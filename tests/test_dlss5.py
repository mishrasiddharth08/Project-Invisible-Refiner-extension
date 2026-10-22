import importlib.util
import json
import sys
import tempfile
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


class FinishedProcess:
    returncode = 0

    def poll(self):
        return self.returncode


class DlssAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = load('dlss5_adapter_test', ROOT / 'lib' / 'dlss5.py')

    def run_file(self, payload, output=b'png'):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.png'
            destination = Path(folder) / 'output.png'
            source.write_bytes(b'input')

            def launch(command, **kwargs):
                self.command = command
                Path(command[4]).write_bytes(output)
                kwargs['stdout'].write((json.dumps(payload) + '\n').encode())
                kwargs['stdout'].flush()
                return FinishedProcess()

            with patch.object(self.adapter, 'executable', return_value=Path(folder) / 'VE_CLI.exe'), \
                 patch.object(self.adapter.subprocess, 'Popen', side_effect=launch):
                result = self.adapter.enhance_file(source, destination, strength=.625, style=2)
            self.assertEqual(destination.read_bytes(), output)
            self.assertEqual(Path(self.command[4]).suffix, destination.suffix)
            self.assertFalse(Path(self.command[4]).exists())
            return result

    def test_strict_json_and_exact_cli_contract(self):
        payload = {'schema_version': 1, 'ok': True, 'completed': 1, 'failed': 0,
                   'items': [{'state': 'complete'}]}
        self.assertEqual(self.run_file(payload), payload)
        self.assertEqual(self.command[1], 'render')
        self.assertEqual(self.command[-6:], ['--style', '2', '--strength', '0.625', '--json', '--quiet'])

    def test_incomplete_json_is_rejected_even_with_an_output_file(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid or incomplete output'):
            self.run_file({'schema_version': 1, 'ok': True, 'completed': 0,
                           'failed': 0, 'items': [{'state': 'complete'}]})

    def test_destination_race_is_not_overwritten_and_temp_is_removed(self):
        payload = {'schema_version': 1, 'ok': True, 'completed': 1, 'failed': 0,
                   'items': [{'state': 'complete'}]}
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.mp4'
            destination = Path(folder) / 'output.mp4'
            source.write_bytes(b'input')

            def launch(command, **kwargs):
                Path(command[4]).write_bytes(b'rendered')
                destination.write_bytes(b'racer')
                kwargs['stdout'].write((json.dumps(payload) + '\n').encode())
                return FinishedProcess()

            with patch.object(self.adapter, 'executable', return_value=Path(folder) / 'VE_CLI.exe'), \
                 patch.object(self.adapter.subprocess, 'Popen', side_effect=launch):
                with self.assertRaisesRegex(ValueError, 'new output filename'):
                    self.adapter.enhance_file(source, destination)
            self.assertEqual(destination.read_bytes(), b'racer')
            self.assertEqual(list(Path(folder).glob('.output-*')), [])

    def test_cancel_removes_partial_temporary_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.mp4'
            destination = Path(folder) / 'output.mp4'
            source.write_bytes(b'input')
            child = Mock(returncode=None)
            child.poll.return_value = None

            def launch(command, **unused):
                Path(command[4]).write_bytes(b'partial')
                return child

            checks = iter((False, True))
            with patch.object(self.adapter, 'executable', return_value=Path(folder) / 'VE_CLI.exe'), \
                 patch.object(self.adapter.subprocess, 'Popen', side_effect=launch), \
                 patch.object(self.adapter, '_stop'):
                with self.assertRaises(self.adapter.Cancelled):
                    self.adapter.enhance_file(source, destination, cancelled=lambda: next(checks))
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(folder).glob('.output-*')), [])

    def test_windows_stop_terminates_only_owned_process_tree(self):
        child = Mock(pid=4321)
        child.poll.return_value = None
        child.wait.return_value = 0
        with patch.object(self.adapter.os, 'name', 'nt'), \
             patch.object(self.adapter.subprocess, 'run') as run:
            self.adapter._stop(child)
        self.assertEqual(Path(run.call_args.args[0][0]).name.lower(), 'taskkill.exe')
        self.assertTrue(Path(run.call_args.args[0][0]).is_absolute())
        self.assertEqual(run.call_args.args[0][1:], ['/PID', '4321', '/T', '/F'])
        self.assertEqual(run.call_args.kwargs['timeout'], 10)
        child.wait.assert_called_once_with(timeout=10)
        child.terminate.assert_not_called()

    def test_image_temp_files_and_alpha_are_preserved(self):
        image = Image.new('RGBA', (3, 2), (12, 34, 56, 0))
        image.putalpha(Image.new('L', image.size, 117))
        image.info['parameters'] = 'base metadata'
        seen = {}

        def enhance(source, destination, **unused):
            seen['source'], seen['destination'] = Path(source), Path(destination)
            with Image.open(source) as decoded:
                self.assertEqual(decoded.mode, 'RGB')
                Image.new('RGB', decoded.size, (90, 80, 70)).save(destination)

        with patch.object(self.adapter, 'enhance_file', side_effect=enhance):
            result = self.adapter.enhance_image(image)
        self.assertEqual(result.mode, 'RGBA')
        self.assertEqual(result.getchannel('A').getextrema(), (117, 117))
        self.assertEqual(result.info['parameters'], 'base metadata')
        self.assertFalse(seen['source'].parent.exists())


class EngineDlssTests(unittest.TestCase):
    def setUp(self):
        modules = types.ModuleType('modules')
        modules.scripts = types.SimpleNamespace(Script=object, AlwaysVisible=True, scripts_data=[])
        modules.script_callbacks = types.SimpleNamespace(on_script_unloaded=Mock(), on_before_ui=Mock(), on_app_started=Mock())
        self.state = types.SimpleNamespace(interrupted=False, skipped=False, stopping_generation=False)
        self.opts = types.SimpleNamespace(data={'sd_model_checkpoint': 'base'})
        modules.shared = types.SimpleNamespace(opts=self.opts, state=self.state)
        modules.processing = types.SimpleNamespace()
        modules.sd_models = types.SimpleNamespace(get_closet_checkpoint_match=Mock())
        sys.modules['modules'] = modules
        self.entry = types.SimpleNamespace(checkpoint_change=Mock(), refresh_model_loading_parameters=Mock())
        sys.modules['modules_forge'] = types.SimpleNamespace(main_entry=self.entry)
        self.memory = types.SimpleNamespace(unload_all_models=Mock(), soft_empty_cache=Mock())
        sys.modules['backend'] = types.SimpleNamespace(memory_management=self.memory)
        self.engine = load('dlss5_engine_test', ROOT / 'scripts' / 'engine.py')
        self.modules = modules
        self.original = Image.new('RGB', (4, 4), 'red')
        self.p = types.SimpleNamespace(
            _pi_refiner=(self.engine._DLSS_TARGET, .25, 0), _pi_refining=False,
            _pi_external_finish=True, image_mask=None, extra_generation_params={})
        self.pp = types.SimpleNamespace(image=self.original, index=0)

    def test_sentinel_session_never_enters_checkpoint_loader(self):
        backend = types.SimpleNamespace(executable=Mock(return_value=Path('VE_CLI.exe')))
        self.engine._dlss_backend = Mock(return_value=backend)
        with self.engine._checkpoint_session(self.p, self.engine._DLSS_TARGET):
            self.opts.data['inside'] = True
        backend.executable.assert_called_once()
        self.entry.checkpoint_change.assert_not_called()
        self.entry.refresh_model_loading_parameters.assert_not_called()
        self.memory.unload_all_models.assert_called_once()
        self.memory.soft_empty_cache.assert_called_once()
        self.assertIsNone(self.p._pi_active_checkpoint)
        self.assertEqual(self.opts.data, {'sd_model_checkpoint': 'base', 'inside': True})

    def test_native_failure_retains_original(self):
        backend = types.SimpleNamespace(executable=Mock(), enhance_image=Mock(side_effect=RuntimeError('native failed')))
        self.engine._dlss_backend = Mock(return_value=backend)
        self.engine.Script().postprocess_image_after_composite(self.p, self.pp)
        self.assertIs(self.pp.image, self.original)
        self.assertEqual(self.p.extra_generation_params['Refiner error'], 'native failed')
        self.memory.unload_all_models.assert_called_once()
        self.memory.soft_empty_cache.assert_called_once()

    def test_cancel_after_native_return_does_not_publish(self):
        replacement = Image.new('RGB', (4, 4), 'blue')

        def enhance(*unused, **kwargs):
            self.state.interrupted = True
            self.assertTrue(kwargs['cancelled']())
            return replacement

        backend = types.SimpleNamespace(executable=Mock(), enhance_image=Mock(side_effect=enhance))
        self.engine._dlss_backend = Mock(return_value=backend)
        self.engine.Script().postprocess_image_after_composite(self.p, self.pp)
        self.assertIs(self.pp.image, self.original)
        self.assertNotIn('Refiner', self.p.extra_generation_params)


if __name__ == '__main__':
    unittest.main()
