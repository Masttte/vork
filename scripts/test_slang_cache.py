"""Cache integrity tests; real compiler checks opt in with VORK_SLANGC."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

spec = importlib.util.spec_from_file_location('slang_cache', Path(__file__).with_name('slang-cache.py'))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class CacheTests(unittest.TestCase):
    def test_contended_lock_serializes_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counter = root / 'counter'
            counter.write_text('0')
            code = """
import importlib.util, pathlib, sys, time
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location('cache', pathlib.Path(sys.argv[1])/'slang-cache.py')
cache = importlib.util.module_from_spec(spec); spec.loader.exec_module(cache)
root = pathlib.Path(sys.argv[2])
for _ in range(20):
    with cache.locked(root/'contended.lock'):
        value = int((root/'counter').read_text())
        time.sleep(.002)
        (root/'counter').write_text(str(value + 1))
"""
            children = [subprocess.Popen([sys.executable, '-c', code, str(Path(__file__).parent), str(root)],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(6)]
            results = [child.communicate(timeout=60) for child in children]
            self.assertEqual([(child.returncode, err.decode()) for child, (_, err) in zip(children, results)],
                             [(0, '')] * 6)
            self.assertEqual(counter.read_text(), '120')

    def test_restore_validates_all_outputs_before_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); entry = root / 'entry'; entry.mkdir()
            outputs = [root / 'code', root / 'reflection']
            for i, value in enumerate((b'code', b'layout')):
                (entry / str(i)).write_bytes(value)
            manifest = dict(key='request', files=[cache.digest(b'code'), cache.digest(b'layout')])
            (entry / 'manifest.json').write_text(json.dumps(manifest))
            self.assertTrue(cache.restore(entry, outputs, 'request'))
            self.assertEqual([p.read_bytes() for p in outputs], [b'code', b'layout'])
            outputs[0].write_bytes(b'previous')
            (entry / '1').write_bytes(b'corrupt')
            self.assertFalse(cache.restore(entry, outputs, 'request'))
            self.assertEqual(outputs[0].read_bytes(), b'previous')
            self.assertFalse(cache.restore(entry, outputs, 'different-request'))

    def test_compiler_identity_includes_libraries_and_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'bin').mkdir(); (root / 'lib').mkdir()
            compiler = root / 'bin/slangc'; compiler.write_bytes(b'compiler')
            library = root / 'lib/libslang.so'; library.write_bytes(b'version one')
            before = cache.toolchain(compiler)
            library.write_bytes(b'version two')
            self.assertNotEqual(before, cache.toolchain(compiler))
            before = cache.toolchain(compiler)
            compiler.write_bytes(b'different compiler')
            self.assertNotEqual(before, cache.toolchain(compiler))

    @unittest.skipUnless(os.environ.get('VORK_SLANGC'), 'requires pinned Slang')
    def test_real_capture_shadowing_relocation_flags_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); store = root / 'store'; compiler = Path(os.environ['VORK_SLANGC'])
            roots = [root / name for name in ('one', 'two')]
            for work in roots:
                work.mkdir(); (work / 'include').mkdir(); (work / 'src').mkdir()
                (work / 'src/main.slang').write_text('import value;\nRWStructuredBuffer<uint> result;\n[numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID) { result[id.x]=answer(); }\n')
                (work / 'include/value.slang').write_text('uint answer() { return 7; }\n')
            def args(work, optimize='-O2'):
                return [str(work/'src/main.slang'), '-I', str(work/'src'), '-I', str(work/'include'), '-target', 'spirv', '-entry', 'main', '-stage', 'compute', optimize, '-o', str(work/'code.spv'), '-reflection-json', str(work/'layout.json')]
            one, two = roots
            self.assertEqual(cache.run(store, compiler, args(one)), 0)
            clean = (one/'code.spv').read_bytes()
            self.assertEqual(cache.run(store, compiler, args(two)), 0)
            events = lambda: [json.loads(line) for line in (store/'events.jsonl').read_text().splitlines()]
            self.assertEqual([e['event'] for e in events()], ['miss', 'hit'])
            self.assertEqual(clean, (two/'code.spv').read_bytes())
            cached_reflection = (two/'layout.json').read_bytes()
            subprocess.run([str(compiler), *args(two)], check=True)
            self.assertEqual(clean, (two/'code.spv').read_bytes())
            self.assertEqual(cached_reflection, (two/'layout.json').read_bytes())
            entry = store/events()[-1]['key']; (entry/'0').write_bytes(b'bad')
            self.assertEqual(cache.run(store, compiler, args(two)), 0)
            self.assertEqual(events()[-1]['event'], 'miss')
            self.assertEqual(clean, (two/'code.spv').read_bytes())
            (two/'src/value.slang').write_text('uint answer() { return 9; }\n')
            self.assertEqual(cache.run(store, compiler, args(two)), 0)
            self.assertEqual(events()[-1]['event'], 'miss')
            self.assertNotEqual(clean, (two/'code.spv').read_bytes())
            self.assertEqual(cache.run(store, compiler, args(two, '-O0')), 0)
            self.assertEqual(events()[-1]['event'], 'miss')
            metal = args(two)
            metal[metal.index('spirv')] = 'metal'
            metal += ['-line-directive-mode', 'none']
            self.assertEqual(cache.run(store, compiler, metal), 0)
            expected = [(two/name).read_bytes() for name in ('code.spv','layout.json')]
            self.assertEqual(cache.run(store, compiler, metal), 0)
            self.assertEqual(events()[-1]['event'], 'hit')
            subprocess.run([str(compiler), *metal], check=True)
            self.assertEqual(expected, [(two/name).read_bytes() for name in ('code.spv','layout.json')])


if __name__ == '__main__':
    unittest.main()
