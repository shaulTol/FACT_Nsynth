import torch
import numpy as np
from scipy import signal

class AudioFourierAugmentation:
    """Applies Fourier-based augmentation to the input audio MFCC."""

    def __init__(self, amplitude=0.5, phase=True):
        self.amplitude = amplitude
        self.phase = phase

    def __call__(self, mfcc):
        # Convert to numpy if tensor
        if isinstance(mfcc, torch.Tensor):
            mfcc_np = mfcc.numpy()
        else:
            mfcc_np = mfcc

        # Apply FFT to each coefficient separately
        augmented_mfcc = np.zeros_like(mfcc_np)

        for i in range(mfcc_np.shape[1]):  # For each MFCC coefficient
            coef = mfcc_np[:, i]

            # Apply FFT
            fft = np.fft.fft(coef)

            # Modify amplitude
            if self.amplitude > 0:
                magnitude = np.abs(fft)
                phase = np.angle(fft)

                # Random amplitude perturbation
                amplitude_factor = 1.0 + np.random.uniform(-self.amplitude, self.amplitude)
                magnitude = magnitude * amplitude_factor

                # Random phase perturbation
                if self.phase:
                    phase_shift = np.random.uniform(-np.pi/4, np.pi/4)
                    phase = phase + phase_shift

                # Reconstruct FFT
                fft_modified = magnitude * np.exp(1j * phase)

                # Inverse FFT
                coef_modified = np.real(np.fft.ifft(fft_modified))
            else:
                coef_modified = coef

            augmented_mfcc[:, i] = coef_modified

        # Convert back to tensor if input was tensor
        if isinstance(mfcc, torch.Tensor):
            return torch.from_numpy(augmented_mfcc).float()

        return augmented_mfcc