bsub -n 16 -W 320 -R "rusage[mem=6144]"  "python main.py --batch 16 --wandb True"
