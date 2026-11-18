"""Content identity for the complete pinned shader compiler distribution."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil


def toolchain_stamp(compiler):
    root = compiler.resolve().parent.parent
    result = []
    for tree in ('bin', 'lib', 'include'):
        for path in sorted([root / tree, *(root / tree).rglob('*')]):
            if '__pycache__' not in path.parts and path.exists():
                stat = path.stat()
                result.append((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return result


def toolchain(compiler):
    # Only the build process supplies this in-memory memo. Never restore stat
    # identities from Actions or a persistent artifact store.
    memo = os.environ.get('VORK_BUILD_COMPILER_IDENTITY_FILE')
    if memo:
        value = json.loads(Path(memo).read_text())
        if value['stamp'] == [list(row) for row in toolchain_stamp(compiler)]:
            return value['digest']

    root = compiler.resolve().parent.parent
    h = hashlib.sha256()
    for tree in ('bin', 'lib', 'include'):
        for path in sorted((root / tree).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                name = path.relative_to(root).as_posix().encode()
                data = path.read_bytes()
                h.update(len(name).to_bytes(8, 'little') + name)
                h.update(len(data).to_bytes(8, 'little') + data)
    return h.hexdigest()



def preparation_identity(binary):
    name = 'slangc.exe' if os.name == 'nt' else 'slangc'
    installed = binary.parent / 'slang/bin' / name
    selected = os.environ.get('VORK_SLANGC') or (str(installed) if installed.is_file() else name)
    compiler = Path(shutil.which(selected) or selected).resolve(strict=True)
    # Explicit loader injection cannot be fingerprinted as a portable compiler.
    # The caller can still prepare without persistence in that environment.
    if any(os.environ.get(key) for key in ('LD_PRELOAD', 'LD_AUDIT', 'DYLD_INSERT_LIBRARIES')):
        raise ValueError('shader preparation reuse does not support loader injection')
    h = hashlib.sha256((platform.platform() + toolchain(compiler)).encode())
    environment = {key: value for key, value in os.environ.items()
                   if key.startswith(('SLANG', 'LD_', 'DYLD_', 'LC_')) or key in ('LANG', 'TZ')}
    h.update(json.dumps(environment, sort_keys=True).encode())
    for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH'):
        for directory in os.environ.get(key, '').split(os.pathsep):
            if not directory:
                continue
            root = Path(directory)
            if root.is_dir():
                for path in sorted(root.iterdir()):
                    if path.is_file() and ('.so' in path.name or path.suffix in ('.dll', '.dylib')):
                        data = path.read_bytes()
                        h.update(path.name.encode() + b'\0' + hashlib.sha256(data).digest())
    return h.hexdigest()
