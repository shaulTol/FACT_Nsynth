import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--domain", "-d", default="sketch", help="Target")
parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
parser.add_argument("--times", "-t", default=1, type=int, help="Repeat times")

args = parser.parse_args()

###############################################################################

source = ["photo", "cartoon", "art_painting", "sketch"]
target = args.domain
source.remove(target)

# Colab paths
input_dir = '/content/AFourier-based-Framework-for-Domain-Generalization/data/datalists'
output_dir = '/content/AFourier-based-Framework-for-Domain-Generalization/train_logs'

config = "PACS/ResNet18"

domain_name = target
path = os.path.join(output_dir, config.replace("/", "_"), domain_name)
##############################################################################

for i in range(args.times):
    # For Colab, we don't need CUDA_VISIBLE_DEVICES as it's handled by Colab
    os.system(f'python /content/AFourier-based-Framework-for-Domain-Generalization/train_colab.py '
              f'--source {source[0]} {source[1]} {source[2]} '
              f'--target {target} '
              f'--input_dir {input_dir} '
              f'--output_dir {output_dir} '
              f'--config {config}')
