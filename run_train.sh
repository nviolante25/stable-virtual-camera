export NCCL_SOCKET_IFNAME=lo
python main.py \
--base configs/example_training/seva-true.yaml \
--wandb \
--projectname seva-on-mvhsamples \
--no-test \
--override_ngpu 0, \
--resume /home/stone/dev/projects/qc_avatar/stable-virtual-camera/logs/2025-07-24T20-27-26_example_training-seva-true