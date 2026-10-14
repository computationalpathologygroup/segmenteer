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
#SBATCH --container-image="dockerdex.umcn.nl:5005#siemdejong/segmenteer"

# segmenteer --help

segmenteer \
    --dataset /dataset \
    --output-dir /outputs/output