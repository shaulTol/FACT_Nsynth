import os
import argparse
import sys

sys.path.append('/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth')

parser = argparse.ArgumentParser()

parser.add_argument("--target", "-t", default="acoustic", help="Target domain")
parser.add_argument("--gpu", "-g", default=0, type=int, help="GPU ID")
parser.add_argument("--batch_size", "-b", default=64, type=int, help="Batch size")
parser.add_argument("--epochs", "-e", default=50, type=int, help="Number of epochs")
parser.add_argument("--max_samples", "-m", default=2000, type=int, help="Max samples per class")

args = parser.parse_args()

# Define all domains
all_domains = ["acoustic", "electronic", "synthetic"]
target_domain = args.target
source_domains = [d for d in all_domains if d != target_domain]

# Paths
output_dir = '/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth/deepall_models'
os.makedirs(output_dir, exist_ok=True)

# Run training
cmd = f'python /content/AFourier-based-Framework-for-Domain-Generalization/Nsynth/train_deepall.py ' \
      f'--source {" ".join(source_domains)} ' \
      f'--target {target_domain} ' \
      f'--batch_size {args.batch_size} ' \
      f'--epochs {args.epochs} ' \
      f'--max_samples {args.max_samples} ' \
      f'--output_dir {output_dir}'

print(f"Running command: {cmd}")
os.system(cmd)