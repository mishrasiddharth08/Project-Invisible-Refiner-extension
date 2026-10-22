"""Local Visual Enhancer CLI adapter; no vendor binaries are redistributed."""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

TARGET = 'DLSS 5 Neural Rendering'
ROOT = Path(__file__).resolve().parents[1]


class Cancelled(RuntimeError):
    pass


def executable():
    configured = os.environ.get('PI_REFINER_DLSS5_HOME')
    config = ROOT / 'dlss5-runtime.json'
    if not configured and config.is_file():
        configured = json.loads(config.read_text(encoding='utf-8')).get('home')
    home = Path(configured).expanduser() if configured else ROOT / 'runtime' / 'Visual Enhancer'
    cli = home if home.name.lower() == 've_cli.exe' else home / 'VE_CLI.exe'
    if not cli.is_file():
        raise RuntimeError('DLSS 5 runtime missing. Set PI_REFINER_DLSS5_HOME to the Visual Enhancer folder.')
    return cli.resolve()


def _stop(child):
    if child.poll() is None:
        if os.name == 'nt':
            try:
                taskkill = str(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'taskkill.exe')
                subprocess.run([taskkill, '/PID', str(child.pid), '/T', '/F'], timeout=10,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=False)
                child.wait(timeout=10)
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def enhance_file(source, destination, strength=1.0, style=0, cancelled=None, timeout=600):
    """Shared image/video file boundary. Each invocation owns its temporal state."""
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError('DLSS 5 strength must be between 0 and 2.')
    if style not in (0, 1, 2):
        raise ValueError('DLSS 5 style must be 0, 1 or 2.')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or source == destination:
        raise ValueError('DLSS 5 requires a new output filename.')
    if cancelled and cancelled():
        raise Cancelled('DLSS 5 cancelled; original retained.')
    cli = executable()
    handle, temporary_name = tempfile.mkstemp(prefix='.' + destination.stem + '-',
                                               suffix=destination.suffix,
                                               dir=str(destination.parent))
    os.close(handle)
    os.unlink(temporary_name)
    temporary = Path(temporary_name)
    command = [str(cli), 'render', str(source), '-o', str(temporary),
               '--style', str(style), '--strength', str(strength), '--json', '--quiet']
    try:
        # File-backed logs avoid deadlocks and unbounded output while polling cancellation.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            child = subprocess.Popen(command, cwd=str(cli.parent), stdout=stdout, stderr=stderr,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            started = time.monotonic()
            try:
                while child.poll() is None:
                    if cancelled and cancelled():
                        raise Cancelled('DLSS 5 cancelled; original retained.')
                    if time.monotonic() - started > timeout:
                        raise RuntimeError('DLSS 5 timed out; original retained.')
                    time.sleep(.1)
                if cancelled and cancelled():
                    raise Cancelled('DLSS 5 cancelled; original retained.')
            finally:
                _stop(child)
            stdout.seek(0, 2)
            stdout.seek(max(0, stdout.tell() - 131072))
            lines = stdout.read().decode('utf-8-sig', errors='replace').splitlines()
            payload = None
            for line in reversed(lines):
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict) and 'schema_version' in candidate:
                        payload = candidate
                        break
                except ValueError:
                    pass
            valid = (payload and payload.get('schema_version') == 1 and payload.get('ok') is True
                     and payload.get('completed') == 1 and payload.get('failed', 0) == 0
                     and len(payload.get('items', [])) == 1
                     and payload['items'][0].get('state') == 'complete')
            if child.returncode or not valid or not temporary.is_file() or not temporary.stat().st_size:
                stderr.seek(0, 2)
                stderr.seek(max(0, stderr.tell() - 3000))
                detail = (payload or {}).get('error') or stderr.read().decode('utf-8', errors='replace')[-1500:]
                raise RuntimeError('DLSS 5 failed: ' + (str(detail).strip() or 'invalid or incomplete output'))
            try:
                os.rename(temporary, destination)  # Windows rename is atomic and refuses an existing target.
            except FileExistsError:
                raise ValueError('DLSS 5 requires a new output filename.')
            return payload
    finally:
        temporary.unlink(missing_ok=True)


def enhance_image(image, strength=1.0, cancelled=None):
    from PIL import Image
    original_info = dict(image.info)
    alpha = image.getchannel('A') if 'A' in image.getbands() else (
        image.convert('RGBA').getchannel('A') if 'transparency' in image.info else None)
    with tempfile.TemporaryDirectory(prefix='pi-dlss5-') as folder:
        source, output = Path(folder) / 'input.png', Path(folder) / 'output.png'
        image.convert('RGB').save(source)
        enhance_file(source, output, strength=strength, cancelled=cancelled)
        with Image.open(output) as decoded:
            decoded.load()
            if decoded.size != image.size:
                raise RuntimeError('DLSS 5 changed image dimensions; original retained.')
            result = decoded.convert('RGB')
        if alpha is not None:
            result.putalpha(alpha)
        result.info.update(original_info)
        return result
