#!/usr/bin/env python3
"""Validated build-time Slang cache. No driver binaries or runtime receipts."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from shader_compiler import toolchain, toolchain_stamp


def digest(data):
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def locked(path):
    with path.open('a+b') as file:
        if os.name == 'nt':
            import msvcrt
            # Windows permits locking beyond EOF. Do not read/write byte zero
            # before acquiring it: another process may already hold that byte.
            while True:
                try:
                    file.seek(0); msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.1)
        else:
            import fcntl
            fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                file.seek(0); msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)


def snapshot(compiler, args):
    includes = [Path(args[i + 1]).resolve() for i, arg in enumerate(args) if arg == '-I']
    native = '-reflection-json' in args
    def label(path):
        path = path.resolve()
        if native:
            for i, root in enumerate(includes):
                if path.is_relative_to(root):
                    return f'<include:{i}>/' + path.relative_to(root).as_posix()
        return str(path)
    with tempfile.TemporaryDirectory(prefix='vork-slang-key-') as temporary:
        work = Path(temporary)
        command = args.copy()
        for flag in ('-o', '-reflection-json'):
            if flag in command:
                index = command.index(flag) + 1
                command[index] = str(work / ('capture-output' + Path(command[index]).suffix))
        repro = work / 'capture.repro'
        for tail in ([*command, '-no-codegen', '-dump-repro', str(repro)], ['-extract-repro', str(repro)]):
            result = subprocess.run([str(compiler), *tail], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode:
                return None
        section = ''; files = []; stamps = []
        for line in (work / 'capture/manifest.txt').read_text().splitlines():
            if line.startswith('['):
                section = line
            elif section == '[files]' and line:
                name, origin = line.split(' -> ', 1)
                path = Path(origin).resolve()
                data = (work / 'capture' / name).read_bytes()
                if any(token in data for token in (b'__FILE__', b'__FILE_NAME__', b'__BASE_FILE__', b'__DATE__', b'__TIME__', b'__TIMESTAMP__')):
                    return None
                stat = path.stat()
                stamp = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if path.read_bytes() != data or not current([stamp]):
                    return None
                stamps.append(stamp)
                files.append((label(path), digest(data)))
        if not files:
            return None
        # Re-capture on every lookup includes new/shadowing imports. Directory
        # stamps additionally reject ABA edits during compilation/publication.
        for root in includes:
            for path in [root, *(p for p in root.rglob('*') if p.is_dir())]:
                stat = path.stat()
                stamps.append((str(path), stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns))
        normalized = []; i = 0
        while i < len(args):
            arg = args[i]
            if native and arg in ('-o', '-reflection-json', '-I'):
                normalized += [arg, '<output>' if arg != '-I' else f'<include:{includes.index(Path(args[i+1]).resolve())}>']
                i += 2
            else:
                normalized.append(label(Path(arg)) if i == 0 else arg)
                i += 1
        return (normalized, sorted(files)), sorted(stamps)


def restore(entry, outputs, key):
    try:
        manifest = json.loads((entry / 'manifest.json').read_text())
        if manifest['key'] != key or len(manifest['files']) != len(outputs):
            return False
        data = [(entry / str(i)).read_bytes() for i in range(len(outputs))]
        if [digest(value) for value in data] != manifest['files']:
            return False
        for output, value in zip(outputs, data):
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as file:
                temporary = Path(file.name); file.write(value)
            os.replace(temporary, output)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def supported(args):
    pairs = {'-I', '-target', '-entry', '-stage', '-warnings-as-errors', '-fp-mode',
             '-line-directive-mode', '-o', '-reflection-json'}
    singles = {'-matrix-layout-row-major', '-zero-initialize', '-O0', '-O1', '-O2', '-O3',
               '-emit-spirv-directly', '-fvk-use-entrypoint-name'}
    if not args or '-target' not in args or '-reflection-json' not in args or '-o' not in args:
        return False
    if any(args.count(flag) != 1 or args.index(flag) + 1 == len(args) for flag in ('-target', '-reflection-json', '-o')):
        return False
    if args[args.index('-target') + 1] not in ('spirv', 'metal'):
        return False
    if '-line-directive-mode' in args and (args.index('-line-directive-mode') + 1 == len(args) or args[args.index('-line-directive-mode') + 1] != 'none'):
        return False
    i = 1
    while i < len(args):
        if any(token in args[i] for token in ('__FILE__', '__DATE__', '__TIME__', '__TIMESTAMP__')):
            return False
        if args[i] in pairs and i + 1 < len(args):
            i += 2
        elif args[i] in singles or args[i].startswith('-D'):
            i += 1
        else:
            return False
    return True


def native_libraries():
    environment = {k: v for k, v in os.environ.items() if k.startswith(('SLANG', 'LD_', 'DYLD_'))}
    # Cargo adds proc-macro directories to the native search path. Those
    # hash-named Rust libraries cannot satisfy a Slang dependency; retain and
    # hash every other library in loader search order, including overrides.
    libraries = []
    for value in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH'):
        for directory in os.environ.get(value, '').split(os.pathsep):
            if not directory:
                continue
            path = Path(directory)
            if path.is_dir():
                files = [(p.name, digest(p.read_bytes())) for p in sorted(path.iterdir())
                         if p.is_file() and ('.so' in p.name or p.suffix in ('.dll', '.dylib'))
                         and not re.fullmatch(r'lib[\w]+-[0-9a-f]{16}\.(so|dylib)', p.name)]
                if files:
                    libraries.append((str(path.resolve()), files))
        environment.pop(value, None)
    if environment:
        return None
    return libraries

def current(stamps):
    for row in stamps:
        try:
            stat = Path(row[0]).stat()
            actual = (row[0], stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if len(row) == 4:
                actual = (row[0], stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns)
            if tuple(row) != actual:
                return False
        except OSError:
            return False
    return True


def run(root, compiler, args):
    started = time.monotonic()
    # Only the renderer's known output-producing commands enter this tool.
    # Unknown/path-sensitive commands still execute normally.
    if not supported(args) or any(os.environ.get(k) for k in ('LD_PRELOAD', 'LD_AUDIT', 'DYLD_INSERT_LIBRARIES')):
        return subprocess.call([str(compiler), *args])
    outputs = [Path(args[args.index(flag) + 1]) for flag in ('-o', '-reflection-json') if flag in args]
    try:
        before = snapshot(compiler, args)
        identity = toolchain(compiler)
    except (OSError, ValueError):
        before = None
    if before is None:
        return subprocess.call([str(compiler), *args])
    libraries = native_libraries()
    if libraries is None:
        return subprocess.call([str(compiler), *args])
    key = digest(json.dumps([digest(Path(__file__).read_bytes()), platform.platform(), platform.machine(), identity, libraries, before[0]], sort_keys=True).encode())
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / (key + '.lock')):
        entry = root / key
        if current(before[1]) and toolchain(compiler) == identity and native_libraries() == libraries and restore(entry, outputs, key):
            os.utime(entry / 'manifest.json', None)
            event = 'hit'; code = 0
        else:
            event = 'miss'
            code = subprocess.call([str(compiler), *args])
            # The capture already resolved every input. File/directory ctime
            # stamps reject concurrent edits without a second compiler parse.
            if code == 0 and current(before[1]) and toolchain(compiler) == identity and native_libraries() == libraries:
                with tempfile.TemporaryDirectory(prefix=key + '.', dir=root) as temporary:
                    staged = Path(temporary) / 'entry'; staged.mkdir()
                    hashes = []
                    for i, output in enumerate(outputs):
                        data = output.read_bytes(); hashes.append(digest(data))
                        (staged / str(i)).write_bytes(data)
                    (staged / 'manifest.json').write_text(json.dumps(dict(key=key, files=hashes)))
                    if entry.exists():
                        shutil.rmtree(entry)
                    staged.rename(entry)
        with locked(root / 'events.lock'), (root / 'events.jsonl').open('a') as log:
            log.write(json.dumps(dict(event=event, key=key, source=args[0], output=str(outputs[0]), args=args, seconds=time.monotonic()-started, time=time.time())) + '\n')
        return code


if __name__ == '__main__':
    if sys.argv[1] == '--identity':
        compiler = Path(sys.argv[2])
        before = toolchain_stamp(compiler)
        identity = toolchain(compiler)
        if before != toolchain_stamp(compiler):
            raise SystemExit('compiler changed while fingerprinting')
        print(json.dumps(dict(digest=identity, stamp=before)))
        raise SystemExit(0)
    raise SystemExit(run(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]))
