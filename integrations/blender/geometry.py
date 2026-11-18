"""Retain Blender material slots, UV identities and named linear colors in glTF.

Aliases share accessors with the original TEXCOORD streams. Surface programs
bind names, so one material also works on meshes with different UV layer order.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import struct

_METADATA = '_vork_uv_bindings'
_BAKE_SOURCE = '_vork_bake_object'


def uv_attribute(name=None):
    return '_VORK_UV_ACTIVE' if name is None else '_VORK_UV_' + name.encode('utf8').hex().upper()


def color_attribute(name):
    return '_VORK_COLOR_' + name.encode('utf8').hex().upper()


@contextmanager
def source_channels():
    """Snapshot object material overrides and named linear colors for glTF.

    Evaluated mesh export can lose object-linked slots whose mesh slot is empty.
    Use private mesh copies so material identities survive without changing other
    objects sharing the original mesh. Named colors use explicit custom streams;
    their identities do not depend on glTF's selected COLOR_n ordering.
    """
    import bpy
    from .source_coordinates import GENERATED, needs_generated, generated_values
    restored, copies = [], []
    generated_objects = set()
    try:
        for obj in bpy.data.objects:
            if obj.type != 'MESH':
                continue
            mesh = obj.data
            slots = [(slot.link, slot.material) for slot in obj.material_slots]
            colors = list(mesh.color_attributes)
            generated = generated_values(obj) if needs_generated(obj) else None
            if generated is not None and GENERATED in mesh.attributes:
                raise ValueError(f'{obj.name}: reserved bridge Generated attribute already exists')
            if generated is None and not colors and not any(link == 'OBJECT' for link, _ in slots):
                continue
            for color in colors:
                if color_attribute(color.name) in mesh.attributes:
                    raise ValueError(f'{obj.name}: reserved bridge color attribute already exists')
            copy = mesh.copy()
            copies.append(copy)
            restored.append((obj, mesh, slots))
            obj.data = copy
            for index, (_, material) in enumerate(slots):
                obj.material_slots[index].link = 'DATA'
                copy.materials[index] = material
            if generated is not None:
                generated_objects.add(obj)
                alias = copy.attributes.new(GENERATED, 'FLOAT_VECTOR', 'POINT')
                alias.data.foreach_set('vector', generated)
            from array import array
            for color in colors:
                values = array('f', [0.]) * (len(color.data) * 4)
                color.data.foreach_get('color', values)
                alias = copy.attributes.new(color_attribute(color.name), 'FLOAT_COLOR', color.domain)
                alias.data.foreach_set('color', values)
        bpy.context.view_layer.update()
        yield generated_objects
    finally:
        for obj, mesh, slots in reversed(restored):
            obj.data = mesh
            for index, (link, material) in enumerate(slots):
                obj.material_slots[index].link = link
                if link == 'OBJECT':
                    obj.material_slots[index].material = material
        for mesh in copies:
            bpy.data.meshes.remove(mesh)
        bpy.context.view_layer.update()


def retain_uv_bindings(document):
    """Consume temporary node metadata and add stable per-primitive aliases."""
    for node in document.get('nodes', []):
        extras = node.get('extras', {})
        metadata = extras.pop(_METADATA, None) if isinstance(extras, dict) else None
        if metadata is None:
            continue
        if not extras:
            node.pop('extras', None)
        names, active, source = json.loads(metadata)
        if source is not None:
            node.setdefault('extras', {})[_BAKE_SOURCE] = source
        if 'mesh' not in node:
            continue
        for primitive in document['meshes'][node['mesh']]['primitives']:
            attributes = primitive['attributes']
            for index, name in enumerate(names):
                accessor = attributes.get(f'TEXCOORD_{index}')
                if accessor is None:
                    continue  # Missing source data stays missing; installation diagnoses a read.
                aliases = [uv_attribute(name)]
                if index == active:
                    aliases.append(uv_attribute())
                for alias in aliases:
                    if alias in attributes and attributes[alias] != accessor:
                        raise ValueError(f'conflicting Blender UV alias {name!r}')
                    attributes[alias] = accessor


def export_gltf(*, bake_sources=False, **options):
    with source_channels() as generated_objects:
        return _export_gltf(bake_sources=bake_sources, generated_objects=generated_objects, **options)


def _export_gltf(*, bake_sources=False, generated_objects=(), **options):
    """Export evaluated geometry with active/named UV identities, without source edits."""
    import bpy
    deps = bpy.context.evaluated_depsgraph_get()
    tagged = []
    from .visibility import METADATA as ray_metadata, flags
    from .instances import SOURCE
    from .source_coordinates import _METADATA as generated_metadata, evaluated_space, retain_evaluated
    try:
        for obj in bpy.data.objects:
            if obj.type not in ('MESH', 'CURVE', 'SURFACE', 'META', 'FONT'):
                continue
            if _METADATA in obj or ray_metadata in obj or generated_metadata in obj:
                raise ValueError(f'object {obj.name!r}: reserved bridge metadata property already exists')
            if bake_sources and _BAKE_SOURCE in obj:
                raise ValueError(f'object {obj.name!r}: reserved bake metadata property already exists')
            tagged.append(obj)
            obj[ray_metadata] = json.dumps(dict(object=obj.name, flags=flags(obj),
                                                 visible_volume_scatter=bool(obj.visible_volume_scatter)))
            if obj.type == 'MESH':
                space = evaluated_space(obj, options.get('export_apply', False))
                if space is not None:
                    obj[generated_metadata] = json.dumps([*space, options.get('export_yup', True)])
                elif obj not in generated_objects:
                    # A Geometry Nodes realization can inherit a reserved field
                    # from a qualified template. It is not that object's ORCO.
                    obj[generated_metadata] = 'null'
                mesh = obj.evaluated_get(deps).data if options.get('export_apply', False) else obj.data
                layers = mesh.uv_layers
                active = next((i for i, layer in enumerate(layers) if layer.active_render), None)
                obj[_METADATA] = json.dumps([[layer.name for layer in layers], active, obj.get(SOURCE, obj.name) if bake_sources else None])
        options['export_extras'] = True
        options['export_attributes'] = True
        result = bpy.ops.export_scene.gltf(**options)
        if 'FINISHED' not in result:
            raise ValueError(f'glTF export did not finish: {result}')
    finally:
        for obj in tagged:
            for key in (_METADATA, ray_metadata, generated_metadata):
                if key in obj: del obj[key]
    path = Path(options['filepath'])
    data = path.read_bytes()
    if data[:4] == b'glTF':
        size, kind = struct.unpack_from('<II', data, 12)
        if kind != 0x4E4F534A:
            raise ValueError('glTF binary has no initial JSON chunk')
        document = json.loads(data[20:20 + size])
        for scene in document.get('scenes', []):
            scene.setdefault('nodes', [])
        retain_uv_bindings(document)
        tail = data[20 + size:]
        if any(generated_metadata in n.get('extras', {}) for n in document.get('nodes', [])):
            if tail:
                length, tag = struct.unpack_from('<II', tail)
                if tag != 0x004E4942:
                    raise ValueError('Generated export requires a glTF binary buffer')
                binary = bytearray(tail[8:8 + length])
                retain_evaluated(document, binary)
                tail = struct.pack('<II', len(binary), tag) + binary + tail[8 + length:]
            else:
                retain_evaluated(document, bytearray())
        encoded = json.dumps(document, separators=(',', ':')).encode('utf8')
        encoded += b' ' * (-len(encoded) % 4)
        path.write_bytes(struct.pack('<III', 0x46546C67, 2, 20 + len(encoded) + len(tail))
                         + struct.pack('<II', len(encoded), kind) + encoded + tail)
    else:
        document = json.loads(data)
        for scene in document.get('scenes', []):
            scene.setdefault('nodes', [])
        retain_uv_bindings(document)
        if any(generated_metadata in n.get('extras', {}) for n in document.get('nodes', [])):
            import base64
            if not document.get('buffers'):
                retain_evaluated(document, bytearray())
                path.write_text(json.dumps(document, separators=(',', ':')) + '\n')
                return
            uri = document['buffers'][0]['uri']
            embedded = uri.startswith('data:')
            binary = bytearray(base64.b64decode(uri.split(',', 1)[1]) if embedded else (path.parent / uri).read_bytes())
            retain_evaluated(document, binary)
            if embedded:
                document['buffers'][0]['uri'] = 'data:application/octet-stream;base64,' + base64.b64encode(binary).decode('ascii')
            else:
                (path.parent / uri).write_bytes(binary)
        path.write_text(json.dumps(document, separators=(',', ':')) + '\n')


def filter_material_overrides(filepath, bindings):
    """Keep overrides present in the exported asset, reporting every missing slot.

    Blender may drop unused slots (or all faces that use one). Such slots cannot
    receive an override. Filter before lowering their graphs, so a dropped,
    unsupported graph cannot prevent the remaining exported scene from loading.
    The renderer still rejects unknown bindings in authored scene documents.
    """
    import warnings
    path = Path(filepath)
    data = path.read_bytes()
    if data[:4] == b'glTF':
        size, kind = struct.unpack_from('<II', data, 12)
        if kind != 0x4E4F534A:
            raise ValueError('glTF binary has no initial JSON chunk')
        document = json.loads(data[20:20 + size])
    else:
        document = json.loads(data)
    materials = document.get('materials', [])
    names = {material['name'] for material in materials if 'name' in material}
    kept = {}
    for key, value in bindings.items():
        # Match Rust usize parsing: a leading '+' is accepted; overflow and
        # nonnumeric suffixes fall back to a literal imported material name.
        index = key[1:].removeprefix('+') if key.startswith('#') else ''
        digits = index.lstrip('0') or '0'
        maximum = str((1 << (8 * struct.calcsize('P'))) - 1)
        numeric = (index.isascii() and index.isdecimal()
                   and (len(digits), digits) <= (len(maximum), maximum))
        present = int(digits) < len(materials) if numeric else key in names
        if key == '*' or present:
            kept[key] = value
        else:
            warnings.warn(f'{path}: material override {key!r} skipped: no matching '
                          'material in exported glTF (Blender may have dropped the slot or its faces)',
                          stacklevel=2)
    return kept
