"""Native Forge Extras entry point for DLSS 5 image enhancement."""
import importlib.util
import inspect
from pathlib import Path
import sys

import gradio as gr

from modules import errors, scripts, scripts_postprocessing, shared


ROOT = Path(__file__).resolve().parents[1]
BACKEND_NAME = '_pi_refiner_dlss5_extras_backend'
_VIDEO_WARNING_SHOWN = False


def _backend():
    loaded = sys.modules.get(BACKEND_NAME)
    if loaded is not None:
        return loaded
    path = ROOT / 'lib' / 'dlss5.py'
    spec = importlib.util.spec_from_file_location(BACKEND_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[BACKEND_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(BACKEND_NAME, None)
        raise
    return module


def _release_gpu(target):
    engine_path = (ROOT / 'scripts' / 'engine.py').resolve()
    for data in scripts.scripts_data:
        if Path(data.path).resolve() == engine_path:
            release = data.script_class.postprocess_image_after_composite.__globals__.get('_release_for_switch')
            if callable(release):
                release(target)
            break
    from backend import memory_management
    memory_management.unload_all_models()
    memory_management.soft_empty_cache()


def _cancelled():
    state = shared.state
    return any(getattr(state, name, False) for name in ('interrupted', 'skipped', 'stopping_generation'))


def _native_video_call():
    return any(frame.function == 'run_postprocessing_video'
               and frame.frame.f_globals.get('__name__') == 'modules.postprocessing'
               for frame in inspect.stack())


class ScriptPostprocessingDLSS5(scripts_postprocessing.ScriptPostprocessing):
    name = 'DLSS 5 Neural Rendering'
    order = 850

    def ui(self):
        enabled = gr.Checkbox(False, label='Enable DLSS 5 Neural Rendering')
        strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label='DLSS 5 intensity')
        gr.Markdown('Enhance existing images. Video support will be added separately.')
        return {'dlss5_enabled': enabled, 'dlss5_strength': strength}

    def process(self, pp, dlss5_enabled=False, dlss5_strength=1.0):
        if not dlss5_enabled:
            return
        global _VIDEO_WARNING_SHOWN
        if _native_video_call():
            if not _VIDEO_WARNING_SHOWN:
                errors.report('DLSS 5 Extras skipped native video; temporal video integration is separate.')
                _VIDEO_WARNING_SHOWN = True
            return
        original = pp.image
        try:
            backend = _backend()
            _release_gpu(backend.TARGET)
            result = backend.enhance_image(original, strength=dlss5_strength, cancelled=_cancelled)
            if _cancelled():
                return
            pp.image = result
            pp.info['DLSS 5'] = f'intensity {float(dlss5_strength):g}'
        except Exception:
            pp.image = original
            errors.report('DLSS 5 Extras failed; original image retained.', exc_info=True)
            warning = getattr(gr, 'Warning', None)
            if callable(warning):
                warning('DLSS 5 failed; original image retained.')
