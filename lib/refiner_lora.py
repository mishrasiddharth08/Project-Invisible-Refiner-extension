"""Keep user LoRA/LoKr adapters in the base stage only."""
from contextlib import contextmanager
import re
import sys

_TAGS = re.compile(r'<(?:lora|lyco|lycoris|lokr):[^>]*>', re.IGNORECASE)


def strip_tags(prompt):
    if isinstance(prompt, list):
        return [strip_tags(value) for value in prompt]
    return _TAGS.sub('', prompt) if isinstance(prompt, str) else prompt


@contextmanager
def final_pass(base, job, shared, sd_models):
    """Suppress the automatic LoRA and clear native patches and conditioning.

    Do not disable extra networks: Forge must activate the empty LoRA set after
    loading a new target too. Disabling that hook retains same-model patches.
    """
    missing = object()
    previous = shared.opts.data.get('sd_lora', missing)
    shared.opts.data['sd_lora'] = 'None'
    job.disable_extra_networks = False
    try:
        for request in (base, job):
            clear = getattr(request, 'clear_prompt_cache', None)
            if callable(clear):
                clear()
        networks = sys.modules.get('networks')
        model_data = getattr(sd_models, 'model_data', None)
        model = getattr(model_data, 'sd_model', None)
        if networks is not None and hasattr(model, 'forge_objects_original'):
            # Also clears current_lora_hash and nunchaku adapter attachments via
            # Forge's own loader, rather than editing weights ourselves.
            networks.load_networks([], [], [])
        yield
    finally:
        if previous is missing:
            shared.opts.data.pop('sd_lora', None)
        else:
            shared.opts.data['sd_lora'] = previous
