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
#SBATCH --container-mounts=/data/pathology/projects:/happiny/projects,/data/pathology/archives:/happiny/archives,/data/pa_cpgarchive:/pa_cpgarchive
#SBATCH --container-image="dockerdex.umcn.nl:5005#siemdejong/segmenteer"

# First we need to make all the model weights available. They are not in the Docker image.
cp -r /pa_cpgarchive/projects/tissue-segmentation/models/ /app

python run.py --input-dir /pa_cpgarchive/projects/tissue-segmentation/data/stains-in-the-wild/hhgm-vanhuizen/HHG/ --output-root /happiny/projects/siemdejong/tissue-segmentation/hhg-output