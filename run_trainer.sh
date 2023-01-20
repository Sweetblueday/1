python3 -m torch.distributed.launch --use_env --nproc_per_node 4 src/main_trainer.py train
# accelerate launch src/main.py train