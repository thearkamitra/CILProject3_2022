bsub -n 2 -W 300 -R "rusage[mem=6000,ngpus_excl_p=1]"  "python main.py --batch 8 --model unet --wandb True --modelname unet_diceloss.pth"
