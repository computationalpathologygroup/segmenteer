#!/bin/bash
#SBATCH --qos=high
#SBATCH --job=segmenteer
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=5
#SBATCH --gpus=1
#SBATCH --nodelist=dlc-tornadus
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --container-mounts=/data/pathology/projects:/happiny/projects,/data/pathology/archives:/happiny/archives,/data/pathology/archives/projects/tissue-segmentation/models:/weights,/data/pa_cpgarchive/projects/tissue-segmentation/data:/pa_cpgarchive/projects/tissue-segmentation/data
#SBATCH --container-image="dockerdex.umcn.nl:5005#siemdejong/segmenteer:pr66"
#SBATCH --no-container-mount-home

# Some methods may need to require downloading, but we don't pin versions in our code.
export SEGMENTEER_ALLOW_UNPINNED_DOWNLOADS=1

# segmenteer --help

segmenteer multi-method \
  --dataset /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/datasets/benchmark-dataset.csv \
  --output-dir /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/outputs/multi-method/benchmark-tissue-masks \
  --configs /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/configs/benchmark-configs.yaml \
  --workers 1 \
  --device cuda \
  --output-format geojson

segmenteer multi-method \
  --dataset /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/datasets/cpg_crowdsource_dataset.csv \
  --output-dir /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/outputs/multi-method/cpg_crowdsource-tissue-masks \
  --configs /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/configs/benchmark-configs.yaml \
  --workers 1 \
  --device cuda \
  --output-format geojson

segmenteer multi-method \
  --dataset /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/datasets/cpg_datateam_dataset.csv \
  --output-dir /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/outputs/multi-method/cpg_datateam-tissue-masks \
  --configs /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/configs/benchmark-configs.yaml \
  --workers 1 \
  --device cuda \
  --output-format geojson

# segmenteer multi-method \
#   --dataset /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/datasets/example-dataset.csv \
#   --output-dir /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/outputs/multi-method/example-tissue-masks \
#   --configs /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/slurm/configs/benchmark-configs.yaml \
#   --workers 4 \
#   --device cuda \
#   --output-format geojson
