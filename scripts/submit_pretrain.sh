#!/bin/bash
#
# Submit this job with:
#   sbatch scripts/submit_pretrain.sh
#
#SBATCH --job-name=lht-pretrain
#SBATCH --partition=gpu_h100_il
#SBATCH --mem=128000mb
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/training_lht_%j.out
#SBATCH --error=logs/training_lht_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load devel/cuda/12.8

# Ensure we run from the submission directory so relative paths resolve
cd "$SLURM_SUBMIT_DIR" || exit

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating it with uv sync..."
    uv sync --extra dev
fi

# Activate uv virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Print job information
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODEID"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Time limit: $SLURM_TIMELIMIT"
echo "Start Time: $(date)"
echo "========================================"

# Print CUDA information
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Print Python and CUDA information
echo "========================================"
echo "Python version:"
python --version
echo "PyTorch CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA not available"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "No GPUs found"
echo "========================================"

# Run the training script
# Default config: configs/pretrain_hierarchical.yaml
# Modify the --config argument if you want to use a different config file
echo "Starting training..."

python scripts/train_pretrain.py --config configs/pretrain_hierarchical.yaml

# To resume from a checkpoint, add:
# --resume-from checkpoints/lht_hierarchical_pretrain/last.ckpt

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
