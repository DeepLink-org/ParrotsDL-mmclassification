#!/bin/bash
set -x
 
# 0. placeholder
workdir=$(cd $(dirname $1); pwd)
if [[ "$workdir" =~ "submodules/mmcls" ]]
then
    if [ -d "$workdir/algolib/configs" ]
    then
        rm -rf $workdir/algolib/configs
        ln -s $workdir/configs $workdir/algolib/
    else
        ln -s $workdir/configs $workdir/algolib/
    fi
else
    if [ -d "$workdir/submodules/mmcls/algolib/configs" ]
    then
        rm -rf $workdir/submodules/mmcls/algolib/configs
        ln -s $workdir/submodules/mmcls/configs $workdir/submodules/mmcls/algolib/
    else
        ln -s $workdir/submodules/mmcls/configs $workdir/submodules/mmcls/algolib/
    fi
fi
 
# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmcls/$3
export PYTORCH_VERSION=1.4
 
# 2. set time
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env
path=$PWD
if [[ "$path" =~ "submodules/mmcls" ]]
then
    pyroot=$path
    comroot=$path/../..
else
    pyroot=$path/submodules/mmcls
    comroot=$path
fi
echo $pyroot
export PYTHONPATH=$comroot:$pyroot:$PYTHONPATH
export FRAME_NAME=mmcls    #customize for each frame
export MODEL_NAME=$3
 
# mmcv path
CONDA_ROOT=/mnt/cache/share/platform/env/miniconda3.6
MMCV_PATH=${CONDA_ROOT}/envs/${CONDA_DEFAULT_ENV}/mmcvs
mmcv_version=1.3.12
export PYTHONPATH=${MMCV_PATH}/${mmcv_version}:$PYTHONPATH
 
 
# init_path
    export PYTHONPATH=$init_path/common/sites/:$PYTHONPATH # necessary for init
 
# 4. build necessary parameter
partition=$1 
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
 
# 5. model choice
export PARROTS_DEFAULT_LOGGER=FALSE


case $MODEL_NAME in
    "ResNet")
        FULL_MODEL="resnet/resnet50_b32x8_imagenet"
        ;;
    "MobileNetV2")
        FULL_MODEL="mobilenet_v2/mobilenet_v2_b32x8_imagenet"
        ;;
    "SEResNet")
        FULL_MODEL="seresnet/seresnet50_b32x8_imagenet"
        ;;
    "ShuffleNetV1")
        FULL_MODEL="shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet"
        ;;
    "ViT")
        FULL_MODEL="vision_transformer/vit-base-p16_pt-64xb64_in1k-224"
        ;;
    "VGG")
        FULL_MODEL="vgg/vgg19bn_b32x8_imagenet"
        ;;
    "ResNeXt")
        FULL_MODEL="resnext/resnext50_32x4d_b32x8_imagenet"
        ;;
    "MobileNet3")
        FULL_MODEL="mobilenet_v3/mobilenet_v3_large_imagenet"
        ;;
    "RegNet")
        FULL_MODEL="regnet/regnetx_4.0gf_b32x8_imagenet"
        ;;
    "RepVGG")
        FULL_MODEL="repvgg/repvgg-A0_4xb64-coslr-120e_in1k"
        ;;
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

set -x

file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
