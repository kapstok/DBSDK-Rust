# How to use the conversion script

## Design

As described in the conversion script itself:


> Limitations by design, because DBM/DBA are simpler than glTF:
>   - Only .glb is supported, not .gltf + external files.
>   - All GLB skins are merged into one DBM skeleton when possible.
>   - Only POSITION/NORMAL/TEXCOORD_0/COLOR_0/JOINTS_0/WEIGHTS_0 are exported.
>   - DBM stores only two bone influences per vertex; this converter keeps the two strongest.
>   - Morph targets, cameras, lights and embedded textures are not exported.
>   - Materials are approximated to the DBM material record.
>   - Animation channels are exported as rest-pose deltas, matching the Blender exporter.
>   - By default GLB UVs are written without V flipping. Use --flip-v if your runtime/old asset needs it.

## Usage

As can be seen from the script's file extension, it's a Python script.

Use `python glb_to_db.py` or `python glb_to_db.py -h` to see how to use it.
