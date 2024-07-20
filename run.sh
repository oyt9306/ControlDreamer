# export PYTHONPATH=$PYTHONPATH:./extern/MV_ControlNet
CFG_PATH=./configs/controldreamer-sd21-shading.yaml
LOADPATH=./outputs/source/Hulk/ckpts/last.ckpt

python launch.py --config ${CFG_PATH} \
    --train --gpu 0 \
    system.prompt_processor.prompt="A high-resolution rendering of an Iron Man, 3d asset" \
    system.geometry_convert_from=${LOADPATH} \
    system.geometry_convert_override.isosurface_threshold=10.