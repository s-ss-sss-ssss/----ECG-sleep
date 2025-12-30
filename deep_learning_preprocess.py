import numpy as np
from scipy.signal import butter, filtfilt

class ECGPreprocessor:
    """
    ECG Preprocessing Pipeline for 30s epochs.
    
    Steps:
    1. Band-pass filter (0.5–40 Hz)
    2. Per-epoch z-score normalization
    3. Amplitude clipping to [-5, 5]
    """

    def __init__(self, sfreq: int = 200):
        self.sfreq = sfreq

    # =========================
    # 1. Band-pass filter
    # =========================
    def bandpass_filter(
        self,
        signal: np.ndarray,
        sfreq: int = 200,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        order: int = 4,
    ):
        """
        Zero-phase Butterworth bandpass filter.
        
        Parameters
        ----------
        signal : np.ndarray
            1D ECG signal
        sfreq : int
            Sampling frequency (Hz)
        lowcut : float
            Low cutoff frequency (Hz)
        highcut : float
            High cutoff frequency (Hz)
        order : int
            Filter order
        
        Returns
        -------
        filtered_signal : np.ndarray
        """
        if sfreq <= 0:
            raise ValueError("Sampling frequency 'sfreq' must be > 0")

        nyq = 0.5 * sfreq
        low = lowcut / nyq
        high = highcut / nyq

        # Validate normalized cutoff frequencies for scipy.signal.butter
        if not (0 < low < 1) or not (0 < high < 1):
            raise ValueError(
                f"Normalized cutoff frequencies must be between 0 and 1 (got low={low}, high={high})"
            )
        if low >= high:
            raise ValueError(f"lowcut must be < highcut (got lowcut={lowcut}, highcut={highcut})")

        b, a = butter(order, [low, high], btype="band")
        filtered_signal = filtfilt(b, a, signal)

        return filtered_signal


    # =========================
    # 2. Per-epoch z-score
    # =========================
    def zscore_normalize(self, signal: np.ndarray, eps: float = 1e-8):
        """
        Per-epoch z-score normalization.
        """
        mean = signal.mean()
        std = signal.std()

        return (signal - mean) / (std + eps)


    # =========================
    # 3. Amplitude clipping
    # =========================
    def amplitude_clipping(self, signal: np.ndarray, clip_value: float = 5.0):
        """
        Clamp signal amplitude to [-clip_value, clip_value].
        """
        return np.clip(signal, -clip_value, clip_value)


    # =========================
    # 4. Full preprocessing pipeline
    # =========================
    def preprocess_ecg_epoch(
        self,
        ecg_epoch: np.ndarray,
        sfreq: int = 200,
    ):
        """
        Preprocess a single 30s ECG epoch.

        Steps:
        1. Band-pass filter (0.5–40 Hz)
        2. Per-epoch z-score normalization
        3. Amplitude clipping

        Parameters
        ----------
        ecg_epoch : np.ndarray
            Shape (6000,)
        sfreq : int
            Sampling frequency

        Returns
        -------
        ecg_processed : np.ndarray
            Shape (6000,)
        """
        assert ecg_epoch.ndim == 1, "Input ECG must be 1D"
        
        # 1. Filter
        ecg_filt = self.bandpass_filter(ecg_epoch, sfreq)

        # 2. Normalize
        ecg_norm = self.zscore_normalize(ecg_filt)

        # 3. Clip
        ecg_clip = self.amplitude_clipping(ecg_norm, clip_value=5.0)

        return ecg_clip.astype(np.float32)
