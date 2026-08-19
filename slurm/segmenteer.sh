#!/bin/bash
#SBATCH --job=segmenteer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --container-mounts=/data/pathology/projects:/happiny/projects,/data/pathology/archives:/happiny/archives,/data/pa_cpgarchive:/pa_cpgarchive
#SBATCH --container-image="dockerdex.umcn.nl:5005#siemdejong/segmenteer"

cp /pa_cpgarchive/projects/tissue-segmentation/models/cpg.onnx /app/models/cpg/cpg.onnx
python run.py
