import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import metal_compile




class UninitialisedLocals(unittest.TestCase):
    def test_finds_block_scope_declarations_without_initialisers(self):
        source = """#include <metal_stdlib>
struct Hit_0
{
    float3 position_0;
};
float3 global_0;
float4 shadow_0(array<float3, int(3)> p_0, const Hit_0 thread* hit_0);
float4 shadow_0(array<float3, int(3)> p_0, const Hit_0 thread* hit_0)
{
    thread array<float3, int(3)> n_0;
    matrix<float,int(4),int(4)>  model_0 = {};
    const Hit_0 thread* pointer_0;
    float set_0 = 1.0f; // float not_code;
    texture2d<float, access::sample> view_0;
    Hit_0 list_0 = { float3(0.0) };
    array<uint, int(2)> threadgroup* shared_0 = context_0->table_0;
    for(int j_0; j_0 < 3; j_0++)
    {
        bool inner_0;
    }
    for(;;)
    {
        return float4(n_0[0], 1.0);
    }
}
"""
        self.assertEqual(metal_compile.uninitialised_locals(source),
                         [(10, 'thread array < float3 , int ( 3 ) > n_0'), (17, 'int j_0'), (19, 'bool inner_0')])
        fixed = source.replace('n_0;', 'n_0 = {};').replace('int j_0;', 'int j_0 = {};').replace('inner_0;', 'inner_0 = {};')
        self.assertEqual(metal_compile.uninitialised_locals(fixed), [])

class WitnessCache(unittest.TestCase):
    def test_exact_inputs_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shader = root / 'entry.metal'
            shader.write_text('kernel void entry() {}')
            cache = root / 'cache'
            calls = []
            fail = False

            def compiler(command, **kwargs):
                calls.append(command)
                if not fail:
                    Path(command[-1]).write_bytes(b'compiled AIR')
                return subprocess.CompletedProcess(command, int(fail), '', 'bad source' if fail else '')

            with patch.dict(os.environ, VORK_METAL_WITNESS_CACHE=str(cache), VORK_METAL_JOBS='2'), \
                    patch.object(metal_compile, 'compiler_identity', return_value=b'compiler-sdk-v1') as identity, \
                    patch.object(metal_compile.subprocess, 'run', side_effect=compiler):
                self.assertEqual(metal_compile.compile_all([shader]), 0)
                self.assertEqual(metal_compile.compile_all([shader]), 0)
                self.assertEqual(len(calls), 1)
                shader.write_text('kernel void changed() {}')
                self.assertEqual(metal_compile.compile_all([shader]), 0)
                self.assertEqual(len(calls), 2)
                identity.return_value = b'compiler-sdk-v2'
                self.assertEqual(metal_compile.compile_all([shader]), 0)
                self.assertEqual(len(calls), 3)
                shader.write_text('broken')
                fail = True
                self.assertEqual(metal_compile.compile_all([shader]), 1)
                self.assertEqual(metal_compile.compile_all([shader]), 1)
                self.assertEqual(len(calls), 5)
                self.assertEqual(len(list(cache.glob('*.air'))), 3)

    def test_parallel_batch_preserves_failure_and_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shaders = [root / f'{i}.metal' for i in range(4)]
            for shader in shaders:
                shader.write_text('float table[];')
            seen = []

            def compiler(command, **kwargs):
                source = Path(command[-3])
                self.assertEqual(source.read_text(), 'float table[1];')
                seen.append(source.name)
                Path(command[-1]).write_bytes(b'AIR')
                return subprocess.CompletedProcess(command, int(source.name == '2.metal'), '', '')

            with patch.dict(os.environ, VORK_METAL_WITNESS_CACHE='', VORK_METAL_JOBS='2'), \
                    patch.object(metal_compile.subprocess, 'run', side_effect=compiler):
                self.assertEqual(metal_compile.compile_all(shaders), 1)
                self.assertCountEqual(seen, [s.name for s in shaders])


if __name__ == '__main__':
    unittest.main()
