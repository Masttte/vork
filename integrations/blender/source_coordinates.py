"""Qualify and retain Blender's source Generated coordinate field."""
import math

GENERATED = '_VORK_1_GENERATED'


def needs_generated(obj):
    seen = set()

    def visit(tree):
        if tree is None or tree.as_pointer() in seen:
            return False
        seen.add(tree.as_pointer())
        for node in tree.nodes:
            if node.type == 'TEX_COORD' and node.outputs['Generated'].is_linked:
                return True
            if node.type.startswith('TEX_') and node.type not in ('TEX_COORD', 'TEX_IMAGE', 'TEX_ENVIRONMENT'):
                vector = node.inputs.get('Vector')
                if vector is not None and not vector.is_linked and not vector.is_unavailable:
                    return True
            if node.type == 'GROUP' and visit(node.node_tree):
                return True
        return False

    return any(visit(slot.material.node_tree) for slot in obj.material_slots
               if slot.material is not None and slot.material.use_nodes)


def generated_values(obj):
    """Return qualified source coordinates, or None to retain the explicit loss.

    Constructive modifiers can synthesize ORCO independently of custom attribute
    interpolation. Unqualified modifier stacks must not advertise exact ORCO.
    """
    if obj.type != 'MESH':
        return None
    mesh = obj.data
    coordinate_mesh = mesh
    from .instances import SOURCE
    if SOURCE in obj:
        import bpy
        source = bpy.data.objects.get(obj[SOURCE])
        if source is None or source == obj or source.type != 'MESH':
            return None
        if source.data.shape_keys is not None or any(m.show_render and m.type not in _CONSTRUCTIVE for m in source.modifiers):
            return None
        coordinate_mesh = source.data
    elif any(m.show_render and m.type != 'SIMPLE_DEFORM' for m in obj.modifiers):
        return None
    if coordinate_mesh.texco_mesh is not None or coordinate_mesh.texture_mesh is not None or (coordinate_mesh.shape_keys is not None and not coordinate_mesh.shape_keys.use_relative):
        return None
    center, size = tuple(coordinate_mesh.texspace_location), tuple(coordinate_mesh.texspace_size)
    if any(not math.isfinite(x) for x in (*center, *size)) or any(x == 0 for x in size):
        return None
    from array import array
    result = array('f')
    for vertex in mesh.vertices:
        result.extend((vertex.co[axis] - center[axis]) / (2 * size[axis]) + .5 for axis in range(3))
    if not all(math.isfinite(x) for x in result):
        return None
    return result

# For these constructive stacks, the independent Cycles oracle equates ORCO
# with evaluated object positions in the original mesh's texture space.
# Deformations and shape keys must instead retain their source field above.
_METADATA = '_vork_generated_space'
_CONSTRUCTIVE = {'SUBSURF', 'BEVEL', 'ARRAY', 'MIRROR'}


def evaluated_space(obj, apply_modifiers):
    if not apply_modifiers or obj.type != 'MESH' or not needs_generated(obj):
        return None
    mesh = obj.data
    modifiers = [m for m in obj.modifiers if m.show_render]
    if not modifiers or any(m.type not in _CONSTRUCTIVE for m in modifiers):
        return None
    if mesh.shape_keys is not None or mesh.texco_mesh is not None or mesh.texture_mesh is not None:
        return None
    center, size = tuple(mesh.texspace_location), tuple(mesh.texspace_size)
    if any(not math.isfinite(x) for x in (*center, *size)) or any(x == 0 for x in size):
        return None
    return center, size


def retain_evaluated(document, binary):
    """Append source Generated fields without changing exported geometry bytes."""
    import copy,json,struct
    meshes = document.get('meshes', [])
    variants, streams = {}, {}
    changed = False
    for node in document.get('nodes', []):
        extras = node.get('extras', {})
        encoded = extras.pop(_METADATA, None) if isinstance(extras, dict) else None
        if encoded is None:
            continue
        if not extras:
            node.pop('extras', None)
        if 'mesh' not in node:
            continue
        space = json.loads(encoded)
        if space is None:
            if not any(GENERATED in p['attributes'] for p in meshes[node['mesh']]['primitives']):
                continue
            key = node['mesh'], None
        else:
            center, size, yup = space
            key = node['mesh'], tuple(center), tuple(size), yup
        if key not in variants:
            mesh = copy.deepcopy(meshes[node['mesh']])
            for primitive in mesh['primitives']:
                attributes = primitive['attributes']
                if space is None:
                    attributes.pop(GENERATED, None)
                    continue
                if GENERATED in attributes:
                    raise ValueError('reserved bridge Generated attribute already exists')
                position = attributes['POSITION']
                stream_key = position, tuple(center), tuple(size), yup
                if stream_key not in streams:
                    accessor = document['accessors'][position]
                    if accessor.get('sparse') or accessor['type'] != 'VEC3' or accessor['componentType'] != 5126:
                        raise ValueError('Generated export requires dense float3 source positions')
                    view = document['bufferViews'][accessor['bufferView']]
                    if view.get('buffer', 0) != 0:
                        raise ValueError('Generated export requires the source export buffer')
                    offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                    stride = view.get('byteStride', 12)
                    payload = bytearray()
                    for i in range(accessor['count']):
                        p = struct.unpack_from('<3f', binary, offset + i * stride)
                        if yup:
                            p = p[0], -p[2], p[1]
                        value = tuple((p[c] - center[c]) / (2 * size[c]) + .5 for c in range(3))
                        if not all(math.isfinite(x) for x in value):
                            raise ValueError('nonfinite source Generated coordinates')
                        payload.extend(struct.pack('<3f', *value))
                    binary.extend(b'\0' * (-len(binary) % 4))
                    target = len(document['bufferViews'])
                    document['bufferViews'].append(dict(buffer=0, byteOffset=len(binary), byteLength=len(payload), target=34962))
                    binary.extend(payload)
                    streams[stream_key] = len(document['accessors'])
                    document['accessors'].append(dict(bufferView=target, componentType=5126, count=accessor['count'], type='VEC3'))
                attributes[GENERATED] = streams[stream_key]
            variants[key] = len(meshes)
            meshes.append(mesh)
        node['mesh'] = variants[key]
        changed = True
    if changed:
        used = sorted({n['mesh'] for n in document.get('nodes', []) if 'mesh' in n})
        remap = {old: new for new, old in enumerate(used)}
        document['meshes'] = [meshes[i] for i in used]
        for node in document.get('nodes', []):
            if 'mesh' in node:
                node['mesh'] = remap[node['mesh']]
        document['buffers'][0]['byteLength'] = len(binary)
    return changed
