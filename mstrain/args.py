import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="resnet50-baseline", help="Name of the Weights & Biases run")
    parser.add_argument("--project", type=str, default="Multi-Scale-CNN", help="Name of the Weights & Biases project")
    parser.add_argument("--entity", type=str, default='pigpeppa', help="Name of the Weights & Biases entity (team or user)")
    parser.add_argument("--offline", action="store_true", help="Run Weights & Biases logger in offline mode")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--max_epochs", type=int, default=90, help="Maximum number of training epochs")
    parser.add_argument("--dataset_path", type=str, default="/mnt/mmtech01/dataset/lzy/ILSVRC2012", help="Path to the ImageNet dataset")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate the model every N epochs")
    parser.add_argument("--trunc", type=float, default=0.01, help="trunc for the si loss")

    args = parser.parse_args()
    return args
