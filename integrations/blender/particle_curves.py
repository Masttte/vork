"""Snapshot evaluated particle paths as existing counted curve geometry.

The Blender conversion operator owns render-child interpolation. Legacy named colors
come from Blender's emitter lookup and are stored once per strand in float
atlases, preserving child interpolation without per-tube-vertex duplication.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import bpy

from .geometry import color_attribute
from .render_geometry import direct_render_objects
from .visibility import flags

# Sort before sparse mesh color streams so large curve meshes do not allocate
# missing color slots preceding this one in the renderer's global stream layout.
COORDINATES = '_VORK_0_HAIR_COORDINATES'


@contextmanager
def converted_path(obj, modifier):
    layer = bpy.context.view_layer
    selected = [(o, o.select_get()) for o in layer.objects]
    active = layer.objects.active
    before = set(bpy.data.objects)
    try:
        for o, _ in selected:
            o.select_set(o == obj)
        layer.objects.active = obj
        result = bpy.ops.object.modifier_convert(modifier=modifier.name)
        created = set(bpy.data.objects) - before
        if 'FINISHED' not in result or len(created) != 1:
            raise ValueError('particle path conversion did not produce one mesh')
        converted, = created
        if converted.type != 'MESH':
            raise ValueError('particle path conversion did not produce a mesh')
        yield converted
    finally:
        for created in set(bpy.data.objects) - before:
            mesh = created.data if created.type == 'MESH' else None
            bpy.data.objects.remove(created, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        for o, state in selected:
            o.select_set(state)
        layer.objects.active = active


def paths(mesh, transform):
    points = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get('co', points.ravel())
    edges = np.empty((len(mesh.edges), 2), np.int32)
    mesh.edges.foreach_get('vertices', edges.ravel())
    if len(mesh.polygons) or not np.all(edges[:, 1] == edges[:, 0] + 1):
        raise ValueError('converted particle paths are not ordered polylines')
    starts_mask = np.ones(len(points), bool)
    starts_mask[edges[:, 1]] = False
    counts = np.diff(np.r_[np.flatnonzero(starts_mask), len(points)])
    # Exact duplicate points make zero-length tube segments. Preserve all other
    # authored points, with no geometric epsilon or density reduction.
    keep = starts_mask.copy()
    if len(points) > 1:
        keep[1:] |= np.any(points[1:] != points[:-1], axis=1)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    counts = np.add.reduceat(keep.astype(np.int64), starts) if len(points) else counts
    points = points[keep]
    valid = counts >= 2
    dropped = int((~valid).sum())
    points, counts = points[np.repeat(valid, counts)], counts[valid]
    if not len(counts):
        return points, counts, np.empty(0), dropped, np.flatnonzero(valid)
    matrix = np.asarray(transform)
    points = points @ matrix[:3, :3].T + matrix[:3, 3]
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    length = np.linalg.norm(np.diff(points, axis=0, prepend=points[:1]), axis=1)
    length[starts] = 0
    cumulative = np.cumsum(length, dtype=np.float64)
    origin = cumulative[starts]
    total = cumulative[starts + counts - 1] - origin
    if np.any(total <= 0) or not np.isfinite(points).all():
        raise ValueError('particle paths contain degenerate or nonfinite curves')
    intercept = (cumulative - np.repeat(origin, counts)) / np.repeat(total, counts)
    return points, counts, intercept, dropped, np.flatnonzero(valid)


def emitter_colors(obj, modifier_name, curve_indices, roots, source_count):
    """Use Blender's legacy byte-color lookup, including interpolated children.

    Legacy particle hair exports corner byte colors, with opaque alpha. RNA
    returns encoded sRGB, so decode after interpolation just as the source
    renderer does. Other color domains/types are not legacy hair attributes.
    """
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    modifier = evaluated.modifiers[modifier_name]
    system = modifier.particle_system
    children = len(system.child_particles)
    first = len(system.particles) if children else 0
    count = children if children else len(system.particles)
    if source_count != count or (len(curve_indices) and int(curve_indices[-1]) >= count):
        raise ValueError('converted paths do not match evaluated particle count')
    # Qualify conversion order against source roots before binding attributes.
    for index in np.linspace(0, len(curve_indices) - 1, min(32, len(curve_indices)), dtype=int):
        point = system.co_hair(evaluated, particle_no=first + int(curve_indices[index]), step=0)
        if not np.allclose(point, roots[index], rtol=1e-5, atol=1e-6):
            raise ValueError('converted particle path order differs from source cache')
    result = {}
    colors = [a for a in evaluated.data.color_attributes
              if a.data_type == 'BYTE_COLOR' and a.domain == 'CORNER']
    for color_index, attr in enumerate(colors):
        values = np.ones((len(curve_indices), 4), np.float32)
        for i, curve in enumerate(curve_indices):
            number = first + int(curve)
            particle = system.particles[0 if children else number]
            values[i, :3] = system.mcol_on_emitter(modifier, particle,
                                      particle_no=number, vcol_no=color_index)
        encoded = values[:, :3]
        values[:, :3] = np.where(encoded <= .04045, encoded / 12.92,
                                  ((encoded + .055) / 1.055) ** 2.4)
        result[color_attribute(attr.name)] = values
    return result


def atlas(writer, key, values, width, height):
    pixels = np.zeros((height * width, 4), np.float32)
    pixels[:len(values)] = values
    image = bpy.data.images.new('__vork_strand_colors__', width=width, height=height,
                                alpha=True, float_buffer=True)
    try:
        image.colorspace_settings.name = 'Linear Rec.709'
        # Blender pixel arrays are bottom-up; the curve lookup uses top-down EXR rows.
        image.pixels.foreach_set(pixels.reshape(height, width, 4)[::-1].ravel())
        return writer.float_image(image, key, ('strand', key),
                                   SimpleNamespace(interpolation='Closest', extension='EXTEND'))
    finally:
        bpy.data.images.remove(image)


def export(scene, writer, *, animated, on_substitution):
    """Call inside gltf_render_settings; return nodes, geometry, buffers, receipts."""
    nodes, geometry, buffers, receipts = [], {}, {}, []
    visible = direct_render_objects(bpy.context.view_layer.layer_collection)
    instanced = set()
    for occurrence in bpy.context.evaluated_depsgraph_get().object_instances:
        if occurrence.is_instance and occurrence.parent is not None and occurrence.parent.original in visible:
            source = occurrence.object.original
            if any(p.settings.type == 'HAIR' and p.settings.render_type == 'PATH' for p in source.particle_systems):
                instanced.add(source.name)
    for name in sorted(instanced):
        on_substitution('instanced particle hair omitted', name + ': only direct emitter snapshots are qualified')
    for obj in sorted(visible, key=lambda o: o.name):
        for modifier in list(obj.modifiers):
            if modifier.type != 'PARTICLE_SYSTEM' or not modifier.show_render:
                continue
            settings = modifier.particle_system.settings
            if settings.type != 'HAIR' or settings.render_type != 'PATH':
                continue
            label = f'{obj.name}/{modifier.name}'
            linear = np.asarray(obj.matrix_world)[:3, :3]
            scales = np.linalg.norm(linear, axis=0)
            gram = linear.T @ linear
            if obj.type != 'MESH' or not np.allclose(gram, np.eye(3) * scales[0] ** 2, rtol=1e-5, atol=1e-8) or scales[0] <= 0:
                on_substitution('particle hair omitted', label + ': requires a mesh emitter with uniform, nonsingular scale')
                continue
            if animated:
                on_substitution('particle hair animation snapshot', label + ': current evaluated frame; strand deformation is not animated')
            slot = settings.material - 1
            if not 0 <= slot < len(obj.material_slots) or obj.material_slots[slot].material is None:
                on_substitution('particle hair omitted', label + ': no assigned source material')
                continue
            with converted_path(obj, modifier) as converted:
                points, counts, intercept, dropped, curve_indices = paths(converted.data, converted.matrix_world)
            if not len(counts):
                on_substitution('particle hair empty', label + ': no nondegenerate render paths')
                continue
            key = f'blender_hair_{len(nodes)}'
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            colors = emitter_colors(obj, modifier.name, curve_indices, points[starts], len(counts) + dropped)
            width = min(1024, len(counts)); height = (len(counts) + width - 1) // width
            images = {name: atlas(writer, f'{key}_color_{i}', values, width, height)
                      for i, (name, values) in enumerate(colors.items())}
            index = np.arange(len(counts))
            uv = np.stack(((index % width + .5) / width, (index // width + .5) / height), axis=1)
            coordinates = np.c_[np.repeat(uv, counts, axis=0), intercept]
            # Source root/tip settings are diameters, matching curve widths.
            factor = 1 - intercept
            if settings.shape < 0:
                factor = factor ** (1 + settings.shape)
            elif settings.shape > 0:
                factor = np.where(factor == 1, 1, 0) if settings.shape == 1 else factor ** (1 / (1 - settings.shape))
            widths = (settings.tip_radius + (settings.root_radius - settings.tip_radius) * factor) * settings.radius_scale * scales[0]
            if settings.use_close_tip:
                widths[starts + counts - 1] = 0
            points = points[:, [0, 2, 1]]; points[:, 2] *= -1
            payload = bytearray()
            for suffix, kind, data, dtype in [('p', 'vec3', points, '<f4'), ('w', 'f32', widths, '<f4'),
                    ('n', 'u32', counts, '<u4'), ('c', 'vec3', coordinates, '<f4')]:
                buffers[key + '_' + suffix] = dict(type=kind, file=key + '.bin', offset=len(payload), count=len(data))
                payload.extend(data.astype(dtype).tobytes())
            (writer.directory / (key + '.bin')).write_bytes(payload)
            geometry[key] = dict(topology='curves', attributes={'POSITION': key + '_p', COORDINATES: key + '_c'},
                                 curve_vertex_counts=key + '_n', widths=key + '_w')
            material = writer.write(obj.material_slots[slot].material,
                         available_attributes={'POSITION', COORDINATES}, attribute_images=images)
            nodes.append(dict(type='mesh', name=key, geometry=key, materials=[material], **flags(obj)))
            if not obj.visible_volume_scatter:
                on_substitution('particle hair ray visibility', label + ': volume-scatter visibility has no independent equivalent')
            receipt = dict(source=label, curves=len(counts), points=len(points), dropped_degenerate=dropped,
                           render_step=settings.render_step, color_source='Blender particle emitter lookup', color_attributes=list(colors),
                           geometry_model='evaluated polylines; open tubes, not native curve intersections')
            receipts.append(receipt)
    return nodes, geometry, buffers, receipts
