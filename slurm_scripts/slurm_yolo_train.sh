#!/bin/bash -l
#SBATCH --job-name=yolo_train
#SBATCH --output=logs/R-%j.out
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:26:10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.schmitt@th-nuernberg.de

mkdir -p logs

BASE_DIR="$WORK/sync/codePort"
DATA_DIR=$TMPDIR

CFG=${1:-"cfg/default.yaml"}
DATA=${2:-"datasets/semmel79.tar"}
SEED=${4:-4040}
# PROJECT="runs"
# NAME="$(basename "${CFG%.*}")-$(basename "${DATA%.*}" | tr '[:upper:]' '[:lower:]')-${SEED}-${SLURM_JOB_ID}"

unset SLURM_EXPORT_ENV

module purge                  # Purge any pre-existing modules
module load python/3.12-conda # Load Python/Conda module
module load cuda/12.6.1       # Load CUDA module for GPU

conda activate conda_ultralytics

# export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
yolo settings wandb=False
yolo settings datasets_dir=$DATA_DIR
yolo settings runs_dir="$BASE_DIR/runs"
yolo settings weights_dir="$BASE_DIR/models"

tar xf "$BASE_DIR/$DATA" --strip-components=1 -C $DATA_DIR
echo ErrorMessage unpacking: $?
echo $DATA_DIR

# [ -e "$BASE_DIR/$CFG" ] && echo "Exists: $BASE_DIR/$CFG" || echo "Missing: $BASE_DIR/$CFG"
# [ -e "$DATA_DIR/dataset.yaml" ] && echo "Exists: $DATA_DIR/dataset.yaml" || echo "Missing: $DATA_DIR/dataset.yaml"
# [ -e "$BASE_DIR/models/yolo11n.pt" ] && echo "Exists: $BASE_DIR/models/yolo11n.pt" || echo "Missing: $BASE_DIR/models/yolo11n.pt"

srun python $BASE_DIR/src/yolo_train.py --cfg $BASE_DIR/$CFG --data "$DATA_DIR/dataset.yaml" --seed $SEED
