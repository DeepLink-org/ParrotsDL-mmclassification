#!/bin/bash
set -x
set -o pipefail

# 0. check the most important SMART_ROOT
echo  "!!!!!SMART_ROOT is" $SMART_ROOT
if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

# 1. set env_path and build soft links for mm configs
if [[ $PWD =~ "mmcls" ]]
then 
    pyroot=$PWD
else
    pyroot=$PWD/mmcls
fi
echo $pyroot
if [ -d "$pyroot/algolib/configs" ]
then
    rm -rf $pyroot/algolib/configs
    ln -s $pyroot/configs $pyroot/algolib/
else
    ln -s $pyroot/configs $pyroot/algolib/
fi

# 2. build file folder for save log and set time
mkdir -p algolib_gen/mmcls/$3
now=$(date +"%Y%m%d_%H%M%S")

# 3. set env variables
export PYTORCH_VERSION=1.4
export PYTHONPATH=$pyroot:$PYTHONPATH
export MODEL_NAME=$3
export FRAME_NAME=mmcls    #customize for each frame
export PARROTS_DEFAULT_LOGGER=FALSE

# 4. init_path
export PYTHONPATH=${SMART_ROOT}:$PYTHONPATH
export PYTHONPATH=${SMART_ROOT}/common/sites:$PYTHONPATH

# 5. build necessary parameter
partition=$1  
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
EXTRA_ARGS=${array[@]:3}
EXTRA_ARGS=${EXTRA_ARGS//--resume/--resume-from}
SRUN_ARGS=${SRUN_ARGS:-""}

# 6. model list
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
    # 该模型存在问题，详见issue: https://jira.sensetime.com/browse/PARROTSXQ-7940
    # "vit-base-p16_pt-64xb64_in1k-224")
    #     FULL_MODEL="vision_transformer/vit-base-p16_pt-64xb64_in1k-224"
    #     ;;
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

# 7. set port and choice model
port=`expr $RANDOM % 10000 + 20000`
file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

# 8. run model
srun -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --cfg-options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
