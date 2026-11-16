"""Experiment-local write guard and deterministic JSON utilities (read-only source adaptation)."""
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXT_ROOT.parent
_AUDIT_INSTALLED = False


def output_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(EXT_ROOT):
        raise ValueError(f'Output escapes new extension: {path}')
    return path


def _descriptor_path(descriptor):
    """Resolve a live Linux descriptor without changing cwd or accepting guesses."""
    try:
        value = os.readlink(f'/proc/self/fd/{int(descriptor)}')
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f'Cannot validate output file descriptor: {descriptor}') from error
    # Anonymous/deleted descriptors have no stable filesystem path to authorize.
    if not os.path.isabs(value) or value.endswith(' (deleted)'):
        raise ValueError(f'Cannot validate anonymous/deleted output descriptor: {descriptor}')
    return Path(value)


def _check_mutation_path(path, dir_fd=None):
    """Audit paths are relative to dir_fd, not necessarily the process cwd.

    CPython reports unspecified dir_fd as -1; AT_FDCWD (-100 on Linux) also
    denotes cwd. Absolute paths ignore dir_fd exactly as the underlying syscall.
    Unknown/deleted descriptors fail closed. Final symlinks are conservatively
    resolved, so this never authorizes writes through a link outside EXT_ROOT.
    """
    if isinstance(path, int):
        return output_path(_descriptor_path(path))
    if not isinstance(path, (str, bytes, os.PathLike)):
        raise ValueError(f'Cannot validate output path type: {type(path).__name__}')
    candidate = Path(os.fsdecode(path))
    if not candidate.is_absolute() and dir_fd not in (None, -1, -100):
        candidate = _descriptor_path(dir_fd) / candidate
    return output_path(candidate)


def _audit_mutation(event, args):
    if event == 'open' and not isinstance(args[0], int):
        mode, flags = args[1], args[2]
        if (mode and any(c in mode for c in 'wax+')) or (flags and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
            # The CPython open audit event omits os.open's dir_fd. Do not infer
            # one or weaken validation: use cwd, failing closed when ambiguous.
            _check_mutation_path(args[0])
    elif event in ('os.remove', 'os.rmdir'):
        _check_mutation_path(args[0], args[1] if len(args)>1 else None)
    elif event in ('os.mkdir', 'os.chmod'):
        _check_mutation_path(args[0], args[2] if len(args)>2 else None)
    elif event == 'os.utime':
        _check_mutation_path(args[0], args[3] if len(args)>3 else None)
    elif event in ('os.rename', 'os.link'):
        _check_mutation_path(args[0], args[2] if len(args)>2 else None)
        _check_mutation_path(args[1], args[3] if len(args)>3 else None)
    elif event == 'os.symlink':
        _check_mutation_path(args[1], args[2] if len(args)>2 else None)


def configure_runtime(guard=True):
    global _AUDIT_INSTALLED
    sys.dont_write_bytecode = True
    for name, suffix in {'XDG_CACHE_HOME':'cache/runtime', 'MPLCONFIGDIR':'cache/mpl',
        'TORCH_HOME':'cache/torch', 'TORCH_EXTENSIONS_DIR':'cache/torch_extensions',
        'HF_HOME':'cache/hf', 'HUGGINGFACE_HUB_CACHE':'cache/hf/hub',
        'NUMBA_CACHE_DIR':'cache/numba', 'CUDA_CACHE_PATH':'cache/cuda',
        'PIP_CACHE_DIR':'cache/pip', 'TMPDIR':'tmp'}.items():
        directory = output_path(EXT_ROOT / suffix)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if guard and not _AUDIT_INSTALLED:
        sys.addaudithook(_audit_mutation)
        _AUDIT_INSTALLED = True


def jsonable(value):
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    return value


def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as file:
        json.dump(jsonable(value), file, indent=2, allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pair_seed(pair_id, seed=0):
    return int.from_bytes(hashlib.sha256(f'{seed}:{pair_id}'.encode()).digest()[:4], 'little') % (2**31-1)


def new_directory(category, label):
    path=output_path(EXT_ROOT/category/(label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]))
    path.mkdir(parents=True, exist_ok=False)
    return path
