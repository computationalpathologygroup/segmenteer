#!/bin/bash
#SBATCH --qos=high
#SBATCH --job=segmenteer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --container-mounts=/data/pathology/projects:/happiny/projects,/data/pathology/archives:/happiny/archives,/data/pathology/archives/projects/tissue-segmentation/models:/weights,/data/pa_cpgarchive/projects/tissue-segmentation/data/benchmark:/pa_cpgarchive/projects/tissue-segmentation/data/benchmark
#SBATCH --container-image="dockerdex.umcn.nl:5005#siemdejong/segmenteer:pr66"
echo $(hostname)
# segmenteer evaluate --help

segmenteer app \
    --output /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/outputs/multi-method/benchmark-tissue-masks/ \
    --data /pa_cpgarchive/projects/tissue-segmentation/data/benchmark/slides \
    --ground-truth /pa_cpgarchive/projects/tissue-segmentation/data/benchmark/geojsons \
    --metrics /happiny/projects/projects/tissue-segmentation/segmenteer/segmenteer-fixes-updates/outputs/multi-method/benchmark-tissue-masks/metrics.csv \
    --host 0.0.0.0