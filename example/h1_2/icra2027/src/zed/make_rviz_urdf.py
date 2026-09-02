#!/usr/bin/env python3
"""make_rviz_urdf.py - derive an RViz-loadable URDF from the H1-2 asset.

h1_2.urdf loads its kinematics fine but RViz shows no robot, because two
things in it are not what urdfdom / RViz want:

  1. every <material> under a <visual> has no name attribute, and urdfdom
     rejects the whole visual element ("Visual material must contain a name
     attribute"), so the links exist but have no geometry to draw;
  2. mesh filenames are relative ("meshes/x.STL"), which RViz cannot resolve
     from a robot_description string.

This writes a copy with named materials and absolute file:// mesh URIs. The
original is left untouched: the FK / pinocchio tooling reads it happily as-is,
and only the RViz path needs these changes.

Usage:
    python3 make_rviz_urdf.py                      # writes h1_2_rviz.urdf
    python3 make_rviz_urdf.py --urdf other.urdf --out /tmp/other_rviz.urdf
"""

import argparse
import os
import re
import sys

DEFAULT_URDF = os.path.expanduser("~/mj_ws/assets/h1_2_description/h1_2.urdf")


def material_name(rgba):
    """Stable, readable name per distinct colour, e.g. '0.1 0.1 0.1 1' ->
    'mat_0p1_0p1_0p1_1'."""
    return "mat_" + "_".join(t.replace(".", "p").replace("-", "m")
                             for t in rgba.split())


def convert(text, mesh_root):
    """Return (converted_text, n_materials_named, n_meshes_rewritten)."""
    # <material> ... <color rgba="..."/> ... </material>  ->  named material
    named = [0]

    def name_material(m):
        named[0] += 1
        return f'<material name="{material_name(m.group(2))}">{m.group(1)}</material>'

    out = re.sub(r'<material>(\s*<color\s+rgba="([^"]*)"\s*/>\s*)</material>',
                 name_material, text)

    # relative mesh path -> absolute file:// URI
    meshes = [0]

    def abs_mesh(m):
        path = m.group(1)
        if "://" in path or os.path.isabs(path):
            return m.group(0)
        meshes[0] += 1
        return f'filename="file://{os.path.join(mesh_root, path)}"'

    out = re.sub(r'filename="([^"]*)"', abs_mesh, out)
    return out, named[0], meshes[0]


def self_test():
    src = ('<visual><geometry><mesh filename="meshes/a.STL"/></geometry>'
           '<material><color rgba="0.1 0.1 0.1 1"/></material></visual>')
    out, n_mat, n_mesh = convert(src, "/root")
    assert n_mat == 1 and n_mesh == 1, (n_mat, n_mesh)
    assert '<material name="mat_0p1_0p1_0p1_1">' in out, out
    assert 'filename="file:///root/meshes/a.STL"' in out, out
    # already absolute / already named are left alone
    out2, n_mat2, n_mesh2 = convert(
        '<material name="keep"><color rgba="1 1 1 1"/></material>'
        '<mesh filename="package://x/y.STL"/>', "/root")
    assert n_mat2 == 0 and n_mesh2 == 0, (n_mat2, n_mesh2)
    assert 'name="keep"' in out2
    print("[self-test] OK - material naming, mesh URI rewrite, idempotence")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--out", default="",
                    help="output path (default: <urdf dir>/h1_2_rviz.urdf)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    if not os.path.isfile(args.urdf):
        sys.exit(f"URDF not found: {args.urdf}")
    root = os.path.dirname(os.path.abspath(args.urdf))
    out_path = args.out or os.path.join(root, "h1_2_rviz.urdf")
    if os.path.abspath(out_path) == os.path.abspath(args.urdf):
        sys.exit("refusing to overwrite the source URDF")

    with open(args.urdf, encoding="utf-8") as f:
        text = f.read()
    converted, n_mat, n_mesh = convert(text, root)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(converted)
    print(f"wrote {out_path}\n  named {n_mat} materials, "
          f"absolutised {n_mesh} mesh paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
