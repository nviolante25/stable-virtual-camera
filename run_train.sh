export NCCL_SOCKET_IFNAME=lo
python main.py \
--base configs/example_training/seva-phase2.yaml \
--wandb \
--projectname seva-on-mvhsamples \
--no-test \
--override_ngpu 0,