# FPO Training and Evaluation Guide

This guide explains how to train and evaluate an FPO (Flow Policy Optimization) agent for the SO100 robot using the LeRobot sim2real framework.

## Prerequisites

- Ensure your `lerobot_sim2real/config/real_robot.py` is properly configured.
- Calibrate your real robot's hardware accurately.
- Complete camera alignment steps as outlined in the main tutorial.
- Make the training script executable:
  ```bash
  chmod +x fpo_run.sh
  ```

## Training FPO Agent

To train the FPO agent, simply run the provided script:

```bash
./fpo_run.sh
```

This script will:
- Train an FPO agent on the SO100GraspCube-v1 environment
- Use the configuration specified in `env_config.json`
- Save checkpoints to `runs/fpo-SO100GraspCube-v1-${seed}/ckpt_x.pt`
- Save evaluation videos to `runs/fpo-SO100GraspCube-v1-${seed}/videos`
- Track progress using Weights and Biases and TensorBoard

You can monitor training progress with TensorBoard:
```bash
tensorboard --logdir runs/
```



## Real World Deployment

1. Place a cube (approximately 2.5cm) in front of the robot within the trained spawn area.
2. Run the evaluation script:
   ```bash
   python lerobot_sim2real/scripts/eval_fpo_rgb.py --env_id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json \
       --checkpoint=path/to/ckpt.pt --no-continuous-eval --control-freq=15
   ```
3. For safety, the script will wait for you to press Enter before each action. Remove `--no-continuous-eval` only if you're confident in the policy's behavior.
4. Always be prepared to press `Ctrl+C` to stop the robot safely.

