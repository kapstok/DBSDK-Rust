#!/usr/bin/env python3
"""
glb_to_db_v6_parentfix.py - Convert binary glTF (.glb) files to DreamBox .dbm/.dba files.

This converter writes the DBM/DBA binary format used by the DreamBox SDK Blender
exporter:

DBM:
  magic "DBM\0", u32 version=1, optional SKEL chunk, one MESH chunk per primitive

DBA:
  magic "DBA\0", u32 version=1, VEC3/QUAT animation chunks

Limitations by design, because DBM/DBA are simpler than glTF:
  - Only .glb is supported, not .gltf + external files.
  - All GLB skins are merged into one DBM skeleton when possible.
  - Only POSITION/NORMAL/TEXCOORD_0/COLOR_0/JOINTS_0/WEIGHTS_0 are exported.
  - DBM stores only two bone influences per vertex; this converter keeps the two strongest.
  - Morph targets, cameras, lights and embedded textures are not exported.
  - Materials are approximated to the DBM material record.
  - Animation channels are exported as rest-pose deltas, matching the Blender exporter.
  - By default GLB UVs are written without V flipping. Use --flip-v if your runtime/old asset needs it.

Typical use:
  python3 glb_to_db.py input.glb -o output.dbm
  python3 glb_to_db.py input.glb -o output.dbm --coords gltf-to-db

The default coordinate conversion is gltf-to-db:
  DB x = -glTF x
  DB y =  glTF y
  DB z = -glTF z
Use --coords identity if your GLB is already in the engine's coordinate system.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------- small math helpers ----------

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # x, y, z, w
Mat4 = List[List[float]]                 # row-major m[row][col]


def mat4_identity() -> Mat4:
    return [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]


def mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def mat4_inverse(m: Mat4) -> Mat4:
    # General 4x4 inverse using Gauss-Jordan elimination.
    a = [[float(m[r][c]) for c in range(4)] + [1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("matrix is not invertible")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        for c in range(8):
            a[col][c] /= div
        for r in range(4):
            if r == col:
                continue
            factor = a[r][col]
            if factor == 0.0:
                continue
            for c in range(8):
                a[r][c] -= factor * a[col][c]
    return [[a[r][c + 4] for c in range(4)] for r in range(4)]


def mat4_from_translation(t: Vec3) -> Mat4:
    m = mat4_identity()
    m[0][3], m[1][3], m[2][3] = t
    return m


def mat4_from_scale(s: Vec3) -> Mat4:
    m = mat4_identity()
    m[0][0], m[1][1], m[2][2] = s
    return m


def mat4_from_quat(q: Quat) -> Mat4:
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return [
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),     0.0],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),     0.0],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy), 0.0],
        [0.0,             0.0,             0.0,             1.0],
    ]


def mat4_from_trs(t: Vec3, r: Quat, s: Vec3) -> Mat4:
    # column-vector convention: M = T * R * S
    return mat4_mul(mat4_mul(mat4_from_translation(t), mat4_from_quat(r)), mat4_from_scale(s))


def mat4_from_gltf_flat(values: Sequence[float]) -> Mat4:
    # glTF matrices are column-major arrays. Convert to row-major m[row][col].
    return [[float(values[c * 4 + r]) for c in range(4)] for r in range(4)]


def mat4_to_flat_row_major(m: Mat4) -> List[float]:
    return [m[r][c] for r in range(4) for c in range(4)]


def vec3_length(v: Vec3) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def vec3_normalize(v: Vec3) -> Vec3:
    l = vec3_length(v)
    if l <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0]/l, v[1]/l, v[2]/l)


def quat_normalize(q: Quat) -> Quat:
    x, y, z, w = q
    l = math.sqrt(x*x + y*y + z*z + w*w)
    if l <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/l, y/l, z/l, w/l)


def quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )


def quat_inverse(q: Quat) -> Quat:
    x, y, z, w = q
    return (-x, -y, -z, w)


def vec3_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    t = max(0.0, min(1.0, t))
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    t = max(0.0, min(1.0, t))
    ax, ay, az, aw = quat_normalize(a)
    bx, by, bz, bw = quat_normalize(b)
    dot = ax*bx + ay*by + az*bz + aw*bw
    if dot < 0.0:
        bx, by, bz, bw = -bx, -by, -bz, -bw
        dot = -dot
    if dot > 0.9995:
        return quat_normalize((
            ax + (bx - ax) * t,
            ay + (by - ay) * t,
            az + (bz - az) * t,
            aw + (bw - aw) * t,
        ))
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_normalize((
        ax * s0 + bx * s1,
        ay * s0 + by * s1,
        az * s0 + bz * s1,
        aw * s0 + bw * s1,
    ))


def quat_from_mat3(m: Mat4) -> Quat:
    # Assumes upper-left 3x3 is a pure rotation matrix.
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return quat_normalize((x, y, z, w))


def decompose_mat4(m: Mat4) -> Tuple[Vec3, Quat, Vec3]:
    t = (m[0][3], m[1][3], m[2][3])
    col0 = (m[0][0], m[1][0], m[2][0])
    col1 = (m[0][1], m[1][1], m[2][1])
    col2 = (m[0][2], m[1][2], m[2][2])
    sx, sy, sz = vec3_length(col0), vec3_length(col1), vec3_length(col2)
    if sx <= 1e-12: sx = 1.0
    if sy <= 1e-12: sy = 1.0
    if sz <= 1e-12: sz = 1.0
    r = mat4_identity()
    for i in range(3):
        r[i][0] = m[i][0] / sx
        r[i][1] = m[i][1] / sy
        r[i][2] = m[i][2] / sz
    q = quat_from_mat3(r)
    return t, q, (sx, sy, sz)


class CoordMapper:
    def __init__(self, mode: str):
        self.mode = mode
        if mode == "identity":
            self.c = mat4_identity()
            self.c_inv = mat4_identity()
            self.qc = (0.0, 0.0, 0.0, 1.0)
        elif mode == "gltf-to-db":
            # 180 degrees around Y: (x,y,z) -> (-x,y,-z)
            self.c = [
                [-1.0, 0.0,  0.0, 0.0],
                [ 0.0, 1.0,  0.0, 0.0],
                [ 0.0, 0.0, -1.0, 0.0],
                [ 0.0, 0.0,  0.0, 1.0],
            ]
            self.c_inv = self.c
            self.qc = (0.0, 1.0, 0.0, 0.0)
        else:
            raise ValueError(f"unknown coordinate mode: {mode}")

    def vec3(self, v: Sequence[float]) -> Vec3:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        if self.mode == "identity":
            return (x, y, z)
        return (-x, y, -z)

    def quat(self, q: Sequence[float]) -> Quat:
        q0 = quat_normalize((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        if self.mode == "identity":
            return q0
        return quat_normalize(quat_mul(quat_mul(self.qc, q0), quat_inverse(self.qc)))

    def mat4(self, m: Mat4) -> Mat4:
        if self.mode == "identity":
            return m
        return mat4_mul(mat4_mul(self.c, m), self.c_inv)


# ---------- glb parsing ----------

COMPONENT_FORMAT = {
    5120: ("b", 1, True),   # BYTE
    5121: ("B", 1, False),  # UNSIGNED_BYTE
    5122: ("h", 2, True),   # SHORT
    5123: ("H", 2, False),  # UNSIGNED_SHORT
    5125: ("I", 4, False),  # UNSIGNED_INT
    5126: ("f", 4, True),   # FLOAT
}
TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def read_glb(path: str) -> Tuple[Dict[str, Any], bytes]:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20:
        raise ValueError("file is too small to be a GLB")
    magic, version, total_len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:  # b'glTF'
        raise ValueError("not a binary glTF/GLB file")
    if version != 2:
        raise ValueError(f"unsupported GLB version: {version}")
    if total_len != len(data):
        raise ValueError("GLB length field does not match file size")

    offset = 12
    json_chunk = None
    bin_chunk = b""
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:  # JSON
            json_chunk = chunk
        elif chunk_type == 0x004E4942:  # BIN\0
            bin_chunk = chunk
    if json_chunk is None:
        raise ValueError("GLB has no JSON chunk")
    doc = json.loads(json_chunk.decode("utf-8"))
    return doc, bin_chunk


@dataclass
class AccessorData:
    values: List[Any]


class GLB:
    def __init__(self, doc: Dict[str, Any], blob: bytes):
        self.doc = doc
        self.blob = blob

    def get(self, key: str) -> List[Any]:
        return self.doc.get(key, [])

    def node_name(self, node_idx: int) -> str:
        node = self.get("nodes")[node_idx]
        return node.get("name", f"node_{node_idx}")

    def local_matrix(self, node_idx: int) -> Mat4:
        node = self.get("nodes")[node_idx]
        if "matrix" in node:
            return mat4_from_gltf_flat(node["matrix"])
        t = tuple(node.get("translation", [0.0, 0.0, 0.0]))  # type: ignore
        r = tuple(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))  # type: ignore
        s = tuple(node.get("scale", [1.0, 1.0, 1.0]))  # type: ignore
        return mat4_from_trs(t, r, s)  # type: ignore

    def scene_roots(self) -> List[int]:
        scenes = self.get("scenes")
        if not scenes:
            return [i for i, _ in enumerate(self.get("nodes"))]
        scene_idx = self.doc.get("scene", 0)
        return list(scenes[scene_idx].get("nodes", []))

    def read_accessor(self, accessor_idx: int, *, raw: bool = False) -> List[Any]:
        accessor = self.get("accessors")[accessor_idx]
        if "sparse" in accessor:
            raise NotImplementedError("sparse accessors are not supported")
        component_type = accessor["componentType"]
        type_name = accessor["type"]
        count = accessor["count"]
        normalized = bool(accessor.get("normalized", False))
        fmt, comp_size, signed = COMPONENT_FORMAT[component_type]
        elem_count = TYPE_COUNTS[type_name]
        elem_size = comp_size * elem_count

        bv_idx = accessor.get("bufferView")
        if bv_idx is None:
            return [tuple(0 for _ in range(elem_count)) if elem_count > 1 else 0 for _ in range(count)]
        bv = self.get("bufferViews")[bv_idx]
        if bv.get("buffer", 0) != 0:
            raise NotImplementedError("only the embedded GLB BIN buffer is supported")
        base = int(bv.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        stride = int(bv.get("byteStride", elem_size))
        unpack_fmt = "<" + fmt * elem_count

        out: List[Any] = []
        for i in range(count):
            pos = base + i * stride
            vals = list(struct.unpack_from(unpack_fmt, self.blob, pos))
            if normalized and not raw:
                vals = [self._normalize_component(v, component_type) for v in vals]
            if elem_count == 1:
                out.append(vals[0])
            else:
                out.append(tuple(vals))
        return out

    @staticmethod
    def _normalize_component(v: int, component_type: int) -> float:
        if component_type == 5120:
            return max(float(v) / 127.0, -1.0)
        if component_type == 5121:
            return float(v) / 255.0
        if component_type == 5122:
            return max(float(v) / 32767.0, -1.0)
        if component_type == 5123:
            return float(v) / 65535.0
        return float(v)


# ---------- DB binary writing ----------

def fixed_str(s: str, n: int) -> bytes:
    b = s.encode("utf-8", errors="replace")[:n]
    return b + bytes(n - len(b))


def pack_mat4_rows(m: Mat4) -> bytes:
    return struct.pack("<" + "f" * 16, *mat4_to_flat_row_major(m))


def pack_half(v: float) -> bytes:
    return struct.pack("<e", float(v))


def clamp_u8(v: float) -> int:
    return max(0, min(255, int(v)))


def scale_vec3(v: Vec3, factor: float) -> Vec3:
    return (v[0] * factor, v[1] * factor, v[2] * factor)


def scale_mat4_translation(m: Mat4, factor: float) -> Mat4:
    """Scale the translation component of a transform matrix.

    This is the correct way to uniformly resize an already-working DBM/DBA
    asset: vertices, rest-pose translations, inverse-bind translations and
    animation delta translations all live in the same object-space units.
    Do NOT shrink every bone's local_scale in the runtime; that scales each
    bone around its own origin and compounds down the hierarchy.
    """
    out = [[float(m[r][c]) for c in range(4)] for r in range(4)]
    out[0][3] *= factor
    out[1][3] *= factor
    out[2][3] *= factor
    return out


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name or "animation"


@dataclass
class SkeletonInfo:
    # joints are stored in DBM export order; bone_index maps glTF node index -> DBM bone id
    joints: List[int]
    roots: List[int]
    children: Dict[int, List[int]]
    bone_index: Dict[int, int]
    inv_bind: Dict[int, Mat4]
    # glTF JOINTS_0 values are indices into the skin.joints array of the mesh node's skin.
    # Keep that original per-skin order so vertex weights can be remapped correctly.
    skin_joints: Dict[int, List[int]]


def build_parent_map(glb: GLB) -> Dict[int, int]:
    parent: Dict[int, int] = {}
    for idx, node in enumerate(glb.get("nodes")):
        for child in node.get("children", []):
            parent[int(child)] = idx
    return parent


def compute_world_matrices(glb: GLB, local_overrides: Optional[Dict[int, Mat4]] = None) -> Dict[int, Mat4]:
    """Return glTF-space world matrices for all nodes.

    DBM only stores exported joints, while glTF skeletons may have ordinary
    transform nodes above those joints. This snowman model uses that pattern:
    e.g. Romp/ArmPerm/Board are non-joint parents that position the actual
    skin skeleton roots. If those parent transforms are not folded into the
    root joints, the rest pose no longer matches the inverse bind matrices and
    the mesh stretches/explodes during skinning.
    """
    local_overrides = local_overrides or {}
    parent = build_parent_map(glb)
    roots = [i for i in range(len(glb.get("nodes"))) if i not in parent]
    world: Dict[int, Mat4] = {}

    def visit(node_idx: int, parent_world: Mat4) -> None:
        local = local_overrides.get(node_idx, glb.local_matrix(node_idx))
        wm = mat4_mul(parent_world, local)
        world[node_idx] = wm
        for child in glb.get("nodes")[node_idx].get("children", []):
            visit(int(child), wm)

    for root in roots:
        visit(root, mat4_identity())
    return world


def nearest_exported_parent(node_idx: int, parent: Dict[int, int], bone_index: Dict[int, int]) -> Optional[int]:
    p = parent.get(node_idx)
    while p is not None:
        if p in bone_index:
            return p
        p = parent.get(p)
    return None


def exported_local_from_world(
    node_idx: int,
    world: Dict[int, Mat4],
    parent: Dict[int, int],
    bone_index: Dict[int, int],
    mapper: CoordMapper,
) -> Mat4:
    """Local matrix as it must be stored in DBM's reduced skeleton tree.

    This collapses non-joint parents into the nearest exported child/root.
    For a root joint this is its full glTF world matrix; for a child joint it is
    relative to the nearest exported parent joint.
    """
    parent_joint = nearest_exported_parent(node_idx, parent, bone_index)
    if parent_joint is None:
        local = world[node_idx]
    else:
        local = mat4_mul(mat4_inverse(world[parent_joint]), world[node_idx])
    return mapper.mat4(local)


def preorder_joint_nodes(root: int, joint_set: set[int], children_all: Dict[int, List[int]], out: List[int]) -> None:
    if root in joint_set:
        out.append(root)
    for child in children_all.get(root, []):
        preorder_joint_nodes(child, joint_set, children_all, out)


def build_skeleton(glb: GLB, mapper: CoordMapper) -> Optional[SkeletonInfo]:
    """Build one DBM skeleton by merging all glTF skins.

    The original version only exported skins[0]. That works for very simple
    models, but this Snowman GLB has five skins: body, two arms, head and
    snowboard. Each mesh node's JOINTS_0 accessor is local to its own skin.joints
    array. If we export only the first skin, or use the wrong skin order for a
    mesh, vertices get bound to the wrong bones and the model explodes.
    """
    skins = glb.get("skins")
    if not skins:
        return None

    parent = build_parent_map(glb)
    children_all: Dict[int, List[int]] = {i: [] for i in range(len(glb.get("nodes")))}
    for child, par in parent.items():
        children_all.setdefault(par, []).append(child)

    # Preserve the original skin-local joint order for vertex JOINTS_0 remapping.
    skin_joints: Dict[int, List[int]] = {}
    all_joint_set: set[int] = set()
    for skin_idx, skin in enumerate(skins):
        joints_original = list(map(int, skin.get("joints", [])))
        if not joints_original:
            continue
        skin_joints[skin_idx] = joints_original
        all_joint_set.update(joints_original)

    if not all_joint_set:
        return None
    if len(all_joint_set) > 256:
        raise ValueError("DBM supports at most 256 bones")

    # Determine roots per skin first. This keeps independent skeleton islands
    # separate instead of accidentally dropping arms/head/board.
    ordered: List[int] = []
    seen: set[int] = set()
    roots: List[int] = []

    def add_preorder(root: int, joint_set: set[int]) -> None:
        if root in joint_set and root not in seen:
            seen.add(root)
            ordered.append(root)
        for child in children_all.get(root, []):
            if child in joint_set:
                add_preorder(child, joint_set)

    for skin_idx, joints_original in skin_joints.items():
        joint_set = set(joints_original)
        skin_roots = [j for j in joints_original if parent.get(j) not in joint_set]
        for r in skin_roots:
            if r not in roots:
                roots.append(r)
            add_preorder(r, joint_set)
        for j in joints_original:
            if j not in seen:
                seen.add(j)
                ordered.append(j)
                if parent.get(j) not in all_joint_set and j not in roots:
                    roots.append(j)

    # Parent/child links must only include exported joints. For merged skins,
    # children can come from any of the skin islands.
    bone_index = {node_idx: i for i, node_idx in enumerate(ordered)}
    children = {j: [c for c in children_all.get(j, []) if c in bone_index] for j in ordered}

    # Read inverse bind matrices per skin. If a joint appears in multiple skins,
    # keep the first value and warn only by being deterministic; duplicate joints
    # should normally have identical bind transforms.
    inv_bind: Dict[int, Mat4] = {}
    for skin_idx, skin in enumerate(skins):
        joints_original = skin_joints.get(skin_idx, [])
        if not joints_original:
            continue
        if "inverseBindMatrices" in skin:
            mats = glb.read_accessor(int(skin["inverseBindMatrices"]))
            for joint_idx, node_idx in enumerate(joints_original):
                if node_idx not in inv_bind:
                    raw = mats[joint_idx]
                    inv_bind[node_idx] = mapper.mat4(mat4_from_gltf_flat(raw))
        else:
            for node_idx in joints_original:
                inv_bind.setdefault(node_idx, mat4_identity())

    for node_idx in ordered:
        inv_bind.setdefault(node_idx, mat4_identity())

    return SkeletonInfo(ordered, roots, children, bone_index, inv_bind, skin_joints)

def write_skel_chunk(glb: GLB, mapper: CoordMapper, skel: SkeletonInfo, *, model_scale: float = 1.0) -> bytes:
    payload = bytearray()

    parent = build_parent_map(glb)
    rest_world = compute_world_matrices(glb)
    rest_export_local: Dict[int, Mat4] = {
        node_idx: exported_local_from_world(node_idx, rest_world, parent, skel.bone_index, mapper)
        for node_idx in skel.joints
    }

    def write_node(node_idx: int) -> None:
        inv = scale_mat4_translation(skel.inv_bind.get(node_idx, mat4_identity()), model_scale)
        local_rest = scale_mat4_translation(rest_export_local[node_idx], model_scale)
        payload.extend(pack_mat4_rows(inv))
        payload.extend(pack_mat4_rows(local_rest))
        payload.extend(struct.pack("<BB", skel.bone_index[node_idx], len(skel.children.get(node_idx, []))))
        for child in skel.children.get(node_idx, []):
            write_node(child)

    for root in skel.roots:
        write_node(root)
    return b"SKEL" + struct.pack("<I", len(payload)) + bytes(payload)


def material_record(glb: GLB, material_idx: Optional[int]) -> bytes:
    if material_idx is None or material_idx < 0 or material_idx >= len(glb.get("materials")):
        return (
            fixed_str("", 32) +
            struct.pack("<BBB", 0, 0, 1) +
            struct.pack("<BBBB", 255, 255, 255, 255) +
            struct.pack("<BBB", 0, 0, 0) +
            struct.pack("<B", 255)
        )
    mat = glb.get("materials")[material_idx]
    name = mat.get("name", f"material_{material_idx}")
    pbr = mat.get("pbrMetallicRoughness", {})
    base = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
    roughness = float(pbr.get("roughnessFactor", 1.0))
    has_tex = 1 if "baseColorTexture" in pbr else 0
    # glTF alphaMode BLEND and MASK both need texture alpha handling in the DB renderer.
    # MASK is used by this snowman atlas: without this the transparent atlas area can render as solid quads.
    alpha_mode = mat.get("alphaMode", "OPAQUE")
    blend = 1 if alpha_mode in ("BLEND", "MASK") else 0
    blend = 0
    # glTF doubleSided=false means backface culling should be enabled.
    cull = 0 if mat.get("doubleSided", False) else 1
    return (
        fixed_str(name, 32) +
        struct.pack("<BBB", has_tex, blend, cull) +
        struct.pack("<BBBB", *(clamp_u8(float(base[i]) * 255.0) for i in range(4))) +
        struct.pack("<BBB", 0, 0, 0) +
        struct.pack("<B", clamp_u8(roughness * 255.0))
    )


def collect_mesh_nodes(glb: GLB) -> List[Tuple[int, Mat4]]:
    result: List[Tuple[int, Mat4]] = []

    def visit(node_idx: int, parent_world: Mat4) -> None:
        local = glb.local_matrix(node_idx)
        world = mat4_mul(parent_world, local)
        node = glb.get("nodes")[node_idx]
        if "mesh" in node:
            result.append((node_idx, local))
        for child in node.get("children", []):
            visit(int(child), world)

    for root in glb.scene_roots():
        visit(root, mat4_identity())
    return result


def read_indices(glb: GLB, primitive: Dict[str, Any], vertex_count: int) -> List[int]:
    if "indices" in primitive:
        return [int(x) for x in glb.read_accessor(int(primitive["indices"]), raw=True)]
    return list(range(vertex_count))


def attr_or_default(glb: GLB, attrs: Dict[str, int], name: str, count: int, default: Any, raw: bool = False) -> List[Any]:
    if name not in attrs:
        return [default for _ in range(count)]
    return glb.read_accessor(int(attrs[name]), raw=raw)


def top_two_weights(
    joints: Sequence[int],
    weights: Sequence[float],
    skel: Optional[SkeletonInfo],
    skin_idx: Optional[int],
) -> Tuple[int, int, int, int]:
    pairs: List[Tuple[float, int]] = []
    skin_joint_nodes: List[int] = []
    if skel is not None and skin_idx is not None:
        skin_joint_nodes = skel.skin_joints.get(int(skin_idx), [])

    for j, w in zip(joints, weights):
        if w <= 0:
            continue
        bone_idx = 0
        # glTF JOINTS_0 values are indices into THIS MESH NODE'S skin.joints,
        # not into the merged/exported DBM skeleton order.
        if skin_joint_nodes and 0 <= int(j) < len(skin_joint_nodes) and skel is not None:
            joint_node = skin_joint_nodes[int(j)]
            bone_idx = skel.bone_index.get(joint_node, 0)
        pairs.append((float(w), bone_idx))

    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = pairs[:2]
    while len(pairs) < 2:
        pairs.append((0.0, 0))
    total = pairs[0][0] + pairs[1][0]
    if total > 0.0:
        w0 = pairs[0][0] / total
        w1 = pairs[1][0] / total
    else:
        w0 = w1 = 0.0
    return clamp_u8(w0 * 255.0), clamp_u8(w1 * 255.0), int(pairs[0][1]) & 0xFF, int(pairs[1][1]) & 0xFF

def write_mesh_chunk(glb: GLB, mapper: CoordMapper, node_idx: int, primitive_idx: int, primitive: Dict[str, Any], skel: Optional[SkeletonInfo], *, flip_v: bool = False, model_scale: float = 1.0) -> bytes:
    attrs = primitive.get("attributes", {})
    node_json = glb.get("nodes")[node_idx]
    skin_idx = node_json.get("skin")
    if "POSITION" not in attrs:
        raise ValueError("mesh primitive without POSITION is not supported")

    positions = glb.read_accessor(int(attrs["POSITION"]))
    vcount = len(positions)
    normals = attr_or_default(glb, attrs, "NORMAL", vcount, (0.0, 1.0, 0.0))
    uvs = attr_or_default(glb, attrs, "TEXCOORD_0", vcount, (0.0, 0.0))
    colors = attr_or_default(glb, attrs, "COLOR_0", vcount, (1.0, 1.0, 1.0, 1.0))
    joints = attr_or_default(glb, attrs, "JOINTS_0", vcount, (0, 0, 0, 0), raw=True)
    weights = attr_or_default(glb, attrs, "WEIGHTS_0", vcount, (0.0, 0.0, 0.0, 0.0))
    indices = read_indices(glb, primitive, vcount)

    if primitive.get("mode", 4) != 4:
        raise ValueError("only TRIANGLES primitives are supported")
    if len(indices) % 3 != 0:
        raise ValueError("triangle index count is not divisible by 3")
    tri_count = len(indices) // 3
    if tri_count > 65535:
        raise ValueError("DBM stores triangle_count as u16; split this mesh first")

    mesh = glb.get("meshes")[glb.get("nodes")[node_idx]["mesh"]]
    mesh_name = mesh.get("name") or glb.node_name(node_idx)
    if len(mesh.get("primitives", [])) > 1:
        mesh_name = f"{mesh_name}_{primitive_idx}"

    local_db = mapper.mat4(glb.local_matrix(node_idx))
    t, r, s = decompose_mat4(local_db)

    payload = bytearray()
    payload.extend(fixed_str(mesh_name, 32))
    t = scale_vec3(t, model_scale)
    payload.extend(struct.pack("<fff", *t))
    payload.extend(struct.pack("<ffff", *r))
    payload.extend(struct.pack("<fff", *s))
    payload.extend(material_record(glb, primitive.get("material")))
    payload.extend(struct.pack("<H", tri_count))

    for idx in indices:
        p = scale_vec3(mapper.vec3(positions[idx]), model_scale)
        n = vec3_normalize(mapper.vec3(normals[idx]))
        col = colors[idx]
        if len(col) == 3:
            col = (col[0], col[1], col[2], 1.0)
        uv = uvs[idx]
        bw0, bw1, bi0, bi1 = top_two_weights(joints[idx], weights[idx], skel, skin_idx)
        payload.extend(pack_half(p[0]) + pack_half(p[1]) + pack_half(p[2]))
        payload.extend(pack_half(n[0]) + pack_half(n[1]) + pack_half(n[2]))
        payload.extend(struct.pack("<BBBB", *(clamp_u8(float(col[i]) * 255.0) for i in range(4))))
        v = 1.0 - float(uv[1]) if flip_v else float(uv[1])
        payload.extend(pack_half(float(uv[0])) + pack_half(v))
        payload.extend(struct.pack("<BBBB", bw0, bw1, bi0, bi1))

    return b"MESH" + struct.pack("<I", len(payload)) + bytes(payload)


def write_dbm(glb: GLB, mapper: CoordMapper, out_path: str, *, flip_v: bool = False, model_scale: float = 1.0) -> Optional[SkeletonInfo]:
    skel = build_skeleton(glb, mapper)
    with open(out_path, "wb") as f:
        f.write(b"DBM\0")
        f.write(struct.pack("<I", 1))
        if skel is not None:
            f.write(write_skel_chunk(glb, mapper, skel, model_scale=model_scale))
        mesh_nodes = collect_mesh_nodes(glb)
        for node_idx, _local in mesh_nodes:
            mesh = glb.get("meshes")[glb.get("nodes")[node_idx]["mesh"]]
            for prim_idx, primitive in enumerate(mesh.get("primitives", [])):
                f.write(write_mesh_chunk(glb, mapper, node_idx, prim_idx, primitive, skel, flip_v=flip_v, model_scale=model_scale))
    return skel


def write_vec3_track(f, channel_id: int, binding_id: int, keys: List[Tuple[float, Vec3]]) -> None:
    f.write(b"VEC3")
    f.write(struct.pack("<I", 12 + 16 * len(keys)))
    f.write(struct.pack("<III", channel_id, binding_id, len(keys)))
    for t, v in keys:
        f.write(struct.pack("<ffff", float(t), float(v[0]), float(v[1]), float(v[2])))


def write_quat_track(f, channel_id: int, binding_id: int, keys: List[Tuple[float, Quat]]) -> None:
    f.write(b"QUAT")
    f.write(struct.pack("<I", 12 + 20 * len(keys)))
    f.write(struct.pack("<III", channel_id, binding_id, len(keys)))
    for t, q in keys:
        f.write(struct.pack("<fffff", float(t), float(q[0]), float(q[1]), float(q[2]), float(q[3])))



def node_default_trs(glb: GLB, node_idx: int) -> Tuple[Vec3, Quat, Vec3]:
    node = glb.get("nodes")[node_idx]
    if "matrix" in node:
        return decompose_mat4(glb.local_matrix(node_idx))
    t = tuple(float(x) for x in node.get("translation", [0.0, 0.0, 0.0]))  # type: ignore[assignment]
    r = quat_normalize(tuple(float(x) for x in node.get("rotation", [0.0, 0.0, 0.0, 1.0])))  # type: ignore[arg-type]
    sc = tuple(float(x) for x in node.get("scale", [1.0, 1.0, 1.0]))  # type: ignore[assignment]
    return t, r, sc  # type: ignore[return-value]


@dataclass
class AnimSamplerData:
    times: List[float]
    values: List[Any]
    interpolation: str
    path: str


def eval_sampler(sampler: AnimSamplerData, time: float, default: Any) -> Any:
    times = sampler.times
    values = sampler.values
    if not times:
        return default
    if time <= times[0]:
        return values[0]
    if time >= times[-1]:
        return values[-1]
    hi = 1
    while hi < len(times) and times[hi] < time:
        hi += 1
    lo = hi - 1
    t0, t1 = times[lo], times[hi]
    if t1 <= t0 or sampler.interpolation == "STEP":
        return values[lo]
    u = (time - t0) / (t1 - t0)
    if sampler.interpolation == "CUBICSPLINE":
        # glTF CUBICSPLINE outputs are triplets: in-tangent, value, out-tangent.
        # Full cubic interpolation is deliberately not implemented here. Use the
        # stored values as a safe fallback, which is usually good enough for simple exports.
        def spline_value(v: Any) -> Any:
            return v[1] if isinstance(v, (list, tuple)) and len(v) == 3 else v
        a = spline_value(values[lo])
        b = spline_value(values[hi])
    else:
        a = values[lo]
        b = values[hi]
    if sampler.path == "rotation":
        return quat_slerp(a, b, u)
    return vec3_lerp(a, b, u)


def make_animated_local_overrides(
    glb: GLB,
    grouped: Dict[int, Dict[str, AnimSamplerData]],
    time_sec: float,
) -> Dict[int, Mat4]:
    """Build glTF local matrices for animated nodes at a specific time."""
    overrides: Dict[int, Mat4] = {}
    for node_idx, node_channels in grouped.items():
        rest_t, rest_r, rest_s = node_default_trs(glb, node_idx)
        t_val = eval_sampler(node_channels["translation"], time_sec, rest_t) if "translation" in node_channels else rest_t
        r_val = eval_sampler(node_channels["rotation"], time_sec, rest_r) if "rotation" in node_channels else rest_r
        s_val = eval_sampler(node_channels["scale"], time_sec, rest_s) if "scale" in node_channels else rest_s
        overrides[node_idx] = mat4_from_trs(t_val, quat_normalize(r_val), s_val)
    return overrides


def write_dbas(glb: GLB, mapper: CoordMapper, skel: Optional[SkeletonInfo], out_dbm_path: str, *, model_scale: float = 1.0, identity_scale_tracks: bool = True) -> List[str]:
    if skel is None:
        return []
    animations = glb.get("animations")
    if not animations:
        return []

    base = os.path.splitext(out_dbm_path)[0]
    written: List[str] = []

    # DBM's skeleton tree contains only exported joints. glTF can have ordinary
    # transform nodes above those joints. Use collapsed export-local matrices for
    # both the rest pose and the animated pose, otherwise inverse bind matrices
    # and animation deltas are calculated in different spaces.
    parent = build_parent_map(glb)
    rest_world = compute_world_matrices(glb)
    rest_local_db: Dict[int, Mat4] = {}
    rest_local_db_inv: Dict[int, Mat4] = {}
    for node_idx in skel.joints:
        rest = scale_mat4_translation(exported_local_from_world(node_idx, rest_world, parent, skel.bone_index, mapper), model_scale)
        rest_local_db[node_idx] = rest
        rest_local_db_inv[node_idx] = mat4_inverse(rest)

    for anim_idx, anim in enumerate(animations):
        name = sanitize_filename(anim.get("name", f"animation_{anim_idx}"))
        out_path = f"{base}_{name}.dba"
        samplers = anim.get("samplers", [])

        # Group glTF channels per exported joint.
        grouped: Dict[int, Dict[str, AnimSamplerData]] = {}
        for channel in anim.get("channels", []):
            target = channel.get("target", {})
            node_idx = target.get("node")
            path = target.get("path")
            if node_idx is None or int(node_idx) not in skel.bone_index:
                continue
            if path not in ("translation", "rotation", "scale"):
                continue
            sampler_json = samplers[int(channel["sampler"])]
            times = [float(x) for x in glb.read_accessor(int(sampler_json["input"]))]
            values = glb.read_accessor(int(sampler_json["output"]))
            values = [tuple(float(c) for c in v) for v in values]
            if len(times) != len(values):
                raise ValueError("animation sampler input/output length mismatch")
            grouped.setdefault(int(node_idx), {})[str(path)] = AnimSamplerData(
                times=times,
                values=values,
                interpolation=str(sampler_json.get("interpolation", "LINEAR")),
                path=str(path),
            )

        if not grouped:
            continue

        tracks: List[Tuple[str, int, int, List[Any]]] = []
        for node_idx, node_channels in grouped.items():
            # Sample at all source key times for this bone.
            times_set = set()
            for sampler in node_channels.values():
                times_set.update(sampler.times)
            times = sorted(times_set)
            if not times:
                continue

            pos_keys: List[Tuple[float, Vec3]] = []
            rot_keys: List[Tuple[float, Quat]] = []
            scale_keys: List[Tuple[float, Vec3]] = []

            for tsec in times:
                overrides = make_animated_local_overrides(glb, grouped, tsec)
                current_world = compute_world_matrices(glb, overrides)
                current_local_db = scale_mat4_translation(
                    exported_local_from_world(node_idx, current_world, parent, skel.bone_index, mapper),
                    model_scale,
                )
                delta = mat4_mul(rest_local_db_inv[node_idx], current_local_db)
                dt, dr, ds = decompose_mat4(delta)
                if identity_scale_tracks:
                    # Keep runtime local_scale at identity unless the asset truly needs
                    # per-bone scaling. This prevents accidental hierarchical shrinking
                    # when the user wants the whole model smaller.
                    ds = (1.0, 1.0, 1.0)

                pos_keys.append((tsec, dt))
                rot_keys.append((tsec, dr))
                scale_keys.append((tsec, ds))

            bone_id = skel.bone_index[node_idx]
            tracks.append(("VEC3", bone_id, 0, pos_keys))
            tracks.append(("QUAT", bone_id, 1, rot_keys))
            tracks.append(("VEC3", bone_id, 2, scale_keys))

        if not tracks:
            continue
        with open(out_path, "wb") as f:
            f.write(b"DBA\0")
            f.write(struct.pack("<I", 1))
            for kind, channel_id, binding_id, keys in tracks:
                if kind == "VEC3":
                    write_vec3_track(f, channel_id, binding_id, keys)  # type: ignore[arg-type]
                else:
                    write_quat_track(f, channel_id, binding_id, keys)  # type: ignore[arg-type]
        written.append(out_path)
    return written

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert GLB to DreamBox DBM/DBA.")
    parser.add_argument("input", help="input .glb file")
    parser.add_argument("-o", "--output", help="output .dbm file; default is input basename + .dbm")
    parser.add_argument("--coords", choices=["gltf-to-db", "identity"], default="gltf-to-db",
                        help="coordinate conversion to apply; default: gltf-to-db")
    parser.add_argument("--no-animations", action="store_true", help="do not write .dba animation files")
    parser.add_argument("--flip-v", action="store_true",
                        help="flip texture V coordinates on export; use this only for older assets/runtimes that expect flipped V")
    parser.add_argument("--model-scale", type=float, default=1.0,
                        help="uniformly scale the exported DBM/DBA asset; use this instead of shrinking every bone's local_scale in Rust")
    parser.add_argument("--keep-scale-tracks", action="store_true",
                        help="keep animation scale channels; by default scale tracks are written as identity to avoid per-bone shrink artifacts")
    args = parser.parse_args(argv)

    in_path = args.input
    out_path = args.output or os.path.splitext(in_path)[0] + ".dbm"
    mapper = CoordMapper(args.coords)

    doc, blob = read_glb(in_path)
    glb = GLB(doc, blob)
    if args.model_scale <= 0.0:
        raise ValueError("--model-scale must be greater than zero")
    skel = write_dbm(glb, mapper, out_path, flip_v=args.flip_v, model_scale=args.model_scale)
    print(f"wrote {out_path}")
    if skel:
        print(f"skeleton bones: {len(skel.joints)}")
        print(f"skins merged: {len(skel.skin_joints)}")
        print(f"model scale: {args.model_scale}")
    else:
        print("no skin/skeleton exported")

    if not args.no_animations:
        dba_paths = write_dbas(glb, mapper, skel, out_path, model_scale=args.model_scale, identity_scale_tracks=not args.keep_scale_tracks)
        for p in dba_paths:
            print(f"wrote {p}")
        if not dba_paths:
            print("no animations exported")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
