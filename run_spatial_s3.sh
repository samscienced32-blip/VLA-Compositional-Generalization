#!/bin/bash
#SBATCH --job-name=openvla_spatial_s3
#SBATCH --partition=gpu-3day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=/nfs_home/users/%u/openvla/logs/slurm_spatial_s3_%j.out
#SBATCH --error=/nfs_home/users/%u/openvla/logs/slurm_spatial_s3_%j.err

SEED=3

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Seed:       $SEED"
echo "Suite:      libero_spatial"
echo "Start:      $(date)"
echo "========================================"

mkdir -p ~/openvla/results
mkdir -p ~/openvla/logs

module purge
module load miniconda/default
module load cuda/12.8

source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla_env

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd ~/openvla

python run_experiment_cluster.py \
    --seed $SEED \
    --cells 1,2,3,4 \
    --num_tasks 10 \
    --rollouts 10

echo "========================================"
echo "Done: $(date)"
echo "========================================"
