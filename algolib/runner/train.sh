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
    init_path=$path/..
else
    pyroot=$path/submodules/mmcls
    comroot=$path
    init_path=$path/submodules
fi
echo $pyroot
export PYTHONPATH=$comroot:$pyroot:$PYTHONPATH
export FRAME_NAME=mmcls    #customize for each frame
export MODEL_NAME=$3
 
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
    "resnet50_b32x8_imagenet")
        FULL_MODEL="resnet/resnet50_b32x8_imagenet"
        ;;
    "mobilenet_v2_b32x8_imagenet")
        FULL_MODEL="mobilenet_v2/mobilenet_v2_b32x8_imagenet"
        ;;
    "seresnet50_b32x8_imagenet")
        FULL_MODEL="seresnet/seresnet50_b32x8_imagenet"
        ;;
    "shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet")
        FULL_MODEL="shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet"
        ;;
    "vit-base-p16_pt-64xb64_in1k-224")
        FULL_MODEL="vision_transformer/vit-base-p16_pt-64xb64_in1k-224"
        ;;
    # 该模型存在精度未对其问题，低了8.23%，详见issue: https://jira.sensetime.com/browse/PARROTSXQ-7942
    # "vgg19bn_b32x8_imagenet")
    #     FULL_MODEL="vgg/vgg19bn_b32x8_imagenet"
    #     ;;
    "resnext50_32x4d_b32x8_imagenet")
        FULL_MODEL="resnext/resnext50_32x4d_b32x8_imagenet"
        ;;
    "mobilenet_v3_large_imagenet")
        FULL_MODEL="mobilenet_v3/mobilenet_v3_large_imagenet"
        ;;
    "regnetx_4.0gf_b32x8_imagenet")
        FULL_MODEL="regnet/regnetx_4.0gf_b32x8_imagenet"
        ;;
    "repvgg-A0_4xb64-coslr-120e_in1k")
        FULL_MODEL="repvgg/repvgg-A0_4xb64-coslr-120e_in1k"
        ;;
    "swin_small_224_b16x64_300e_imagenet")
        FULL_MODEL="swin_transformer/swin_small_224_b16x64_300e_imagenet"
        ;;
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

port=`expr $RANDOM % 10000 + 20000`

set -x

file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
