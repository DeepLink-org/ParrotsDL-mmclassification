port=`expr $RANDOM % 10000 + 20000`
time=$(date "+%Y%m%d%H%M%S")
# srun -p camb_mlu290 -n8 --gres=mlu:8 --ntasks-per-node 8 --job-name=shufflenet python tools/train.py configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py --launcher=slurm --cfg-options dist_params.port=$port
# srun -p camb_mlu290 -n1 --gres=mlu:1 --ntasks-per-node 1 --job-name=shufflenet python tools/train.py configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py --launcher=slurm --cfg-options dist_params.port=$port

nohup srun -p camb_mlu290 -n8 --gres=mlu:8 --ntasks-per-node 8 --job-name=shufflenet python tools/train.py configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py --resume-from=work_dirs/shufflenet-v2-1x_16xb64_in1k/latest.pth --launcher=slurm --cfg-options dist_params.port=$port >log/shurffle_net_${time}.log 2>&1 &