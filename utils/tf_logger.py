# Code updated from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import io

class TFLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Use PyTorch's SummaryWriter which is compatible with TensorBoard
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        # PyTorch expects images in a different format, so we convert them
        if isinstance(images, list):
            # If it's a list of images, convert to tensor
            if len(images) > 0:
                try:
                    # Add a batch dimension if needed
                    images_tensor = np.stack(images)
                    self.writer.add_images(tag, images_tensor, step)
                except Exception as e:
                    print(f"Warning: Failed to add images to tensorboard: {e}")
                    # Try to add them individually as fallback
                    for i, img in enumerate(images):
                        try:
                            # Convert image to tensor and add
                            self.writer.add_image(f'{tag}/{i}', img, step)
                        except:
                            print(f"Warning: Failed to add image {i} to tensorboard")
        else:
            # Handle single image
            try:
                self.writer.add_image(tag, images, step)
            except Exception as e:
                print(f"Warning: Failed to add image to tensorboard: {e}")

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        try:
            self.writer.add_histogram(tag, values, step, bins=bins)
        except Exception as e:
            print(f"Warning: Failed to add histogram to tensorboard: {e}")

    def flush(self):
        """Flush the writer."""
        self.writer.flush()

    def close(self):
        """Close the writer."""
        self.writer.close()