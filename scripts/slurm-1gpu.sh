#!/bin/bash
#SBATCH --job-name=4848_svis
#SBATCH --partition=ampere
#SBATCH --account=neutrino:icarus-ml
#SBATCH --output=logs/output_4848_1gpu_%j.log
#SBATCH --error=logs/error_4848_1gpu_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=15:00:00
#SBATCH --gpu-bind=none


echo 'Copying plib to lscratch'
mkdir -p /lscratch/youngsam/tmp/plib
cp /sdf/home/y/youngsam/sw/dune/siren-t/data/plib_2x2_module0_06052024_4848.h5 \
   /lscratch/youngsam/tmp/plib

srun singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch \
                    /fs/ddn/sdf/group/neutrino/images/develop.sif \
                    python /sdf/home/y/youngsam/sw/dune/siren-t/train_multistage.py \
                    --config /sdf/home/y/youngsam/sw/dune/siren-t/config/siren_4848-slurm.yaml
                    
echo 'Removing plib from lscratch'
rm -r /lscratch/youngsam/tmp/plib