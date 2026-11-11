#!/bin/bash
#SBATCH --qos=high
#SBATCH --job=segmenteer
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=5
#SBATCH --gpus=1
#SBATCH --nodelist=dlc-meowth,dlc-groudon,dlc-slowpoke,dlc-arceus
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --container-mounts=/data/pathology/projects:/happiny/projects,/data/pathology/archives:/happiny/archives,/data/pathology/archives/projects/tissue-segmentation/models:/weights,/data/pa_cpgarchive/projects/tissue-segmentation/data/stains-in-the-wild/hhgm-vanhuizen/HHG/:/dataset,/data/pathology/projects/siemdejong/tissue-segmentation/hhg-output:/outputs
#SBATCH --container-image="dockerdex.umcn.nl:5005#siemdejong/segmenteer:pr66"

# segmenteer --help

# segmenteer \
#     --dataset /dataset \
#     --output-dir /outputs/output

segmenteer multi-method \
  --dataset /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/datasets/benchmark-dataset.csv \
  --output-dir /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/outputs/multi-method/benchmark-tissue-masks \
  --configs /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/configs/benchmark-configs.yaml \
  --workers 1 \
  --device cuda:0 \
  --output-format geojson
