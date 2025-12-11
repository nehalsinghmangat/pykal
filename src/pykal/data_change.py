"""
Data corruption and preparation utilities for simulating and handling
real-world hardware sensor issues in robotics applications.
"""

from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray


class corrupt:
    """
    Simulate common hardware data corruption issues.

    All methods are static and take NumPy arrays as input, returning
    corrupted versions. Useful for testing robustness of estimators
    and controllers before hardware deployment.
    """

    @staticmethod
    def with_gaussian_noise(
        data: NDArray,
        std: Optional[float] = None,
        mean: Optional[float] = None,
        Q: Optional[NDArray] = None,
        seed: Optional[int] = None
    ) -> NDArray:
        """
        Add Gaussian (normal) noise to data.

        Common in analog sensors due to thermal noise, quantization, etc.
        Supports both scalar and multivariate (correlated) noise.

        Parameters
        ----------
        data : NDArray
            Input data array of shape (n,) for n-dimensional data
        std : float, optional
            Standard deviation of noise for scalar case (default 0.1 if Q not provided)
        mean : float, optional
            Mean of noise distribution for scalar case (default 0.0 if Q not provided)
        Q : NDArray, optional
            Covariance matrix of shape (n, n) for multivariate noise.
            If provided, std and mean are ignored. Each element can have
            different noise characteristics with correlations.
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        NDArray
            Data with added Gaussian noise

        Examples
        --------
        Scalar noise (independent, same std for all elements):

        >>> import numpy as np
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> noisy = corrupt.with_gaussian_noise(data, std=0.1, seed=42)
        >>> noisy.shape == data.shape
        True
        >>> np.abs(noisy - data).mean() < 0.5  # noise is bounded
        True

        Multivariate noise with covariance matrix:

        >>> data = np.array([1.0, 2.0, 3.0])
        >>> Q = np.array([[0.1, 0.02, 0.0],
        ...               [0.02, 0.2, 0.01],
        ...               [0.0, 0.01, 0.15]])
        >>> noisy = corrupt.with_gaussian_noise(data, Q=Q, seed=42)
        >>> noisy.shape == data.shape
        True
        >>> np.abs(noisy - data).mean() < 1.0  # noise is bounded
        True

        Different variance for each element (diagonal covariance):

        >>> data = np.array([1.0, 2.0, 3.0])
        >>> Q = np.diag([0.01, 0.1, 0.5])  # different noise per element
        >>> noisy = corrupt.with_gaussian_noise(data, Q=Q, seed=42)
        >>> noisy.shape == data.shape
        True
        """
        rng = np.random.default_rng(seed)

        if Q is not None:
            # Multivariate case with covariance matrix
            if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
                raise ValueError("Q must be a square matrix")
            if Q.shape[0] != len(data):
                raise ValueError(f"Q shape {Q.shape} incompatible with data length {len(data)}")

            # Generate multivariate Gaussian noise with zero mean
            noise = rng.multivariate_normal(np.zeros(len(data)), Q)
        else:
            # Scalar case - independent noise with same std for all elements
            std_val = std if std is not None else 0.1
            mean_val = mean if mean is not None else 0.0
            noise = rng.normal(mean_val, std_val, size=data.shape)

        return data + noise

    @staticmethod
    def with_bounce(
        data: NDArray,
        duration: int = 3,
        amplitude: float = 0.5,
        seed: Optional[int] = None
    ) -> NDArray:
        """
        Simulate contact bounce on digital/binary signals.

        Common in switches, encoders, limit switches. Creates rapid
        oscillations when signal changes state.

        Parameters
        ----------
        data : NDArray
            Input data array (typically binary or step changes)
        duration : int, optional
            Number of samples to bounce (default 3)
        amplitude : float, optional
            Amplitude of bounce oscillation (default 0.5)
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        NDArray
            Data with bounce artifacts at transitions

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([0., 0., 1., 1., 1.])
        >>> bounced = corrupt.with_bounce(data, duration=2, seed=42)
        >>> bounced.shape == data.shape
        True
        """
        rng = np.random.default_rng(seed)
        result = data.copy()

        # Find transitions (where diff is non-zero)
        if len(data) > 1:
            diff = np.diff(data)
            transitions = np.where(np.abs(diff) > 1e-6)[0]

            for trans_idx in transitions:
                # Add oscillation after transition
                end_idx = min(trans_idx + duration + 1, len(result))
                bounce_len = end_idx - (trans_idx + 1)
                if bounce_len > 0:
                    oscillation = amplitude * rng.choice([-1, 1], size=bounce_len)
                    result[trans_idx + 1:end_idx] += oscillation

        return result

    @staticmethod
    def with_dropouts(
        data: NDArray,
        dropout_rate: float = 0.1,
        fill_value: float = np.nan,
        seed: Optional[int] = None
    ) -> NDArray:
        """
        Randomly drop data points (packet loss, sensor failures).

        Common in wireless communication, intermittent connections.

        Parameters
        ----------
        data : NDArray
            Input data array
        dropout_rate : float, optional
            Fraction of data points to drop (default 0.1)
        fill_value : float, optional
            Value to use for dropped points (default np.nan)
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        NDArray
            Data with random dropouts

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1., 2., 3., 4., 5.])
        >>> dropped = corrupt.with_dropouts(data, dropout_rate=0.3, seed=42)
        >>> dropped.shape == data.shape
        True
        >>> np.isnan(dropped).sum() > 0  # some data dropped
        True
        """
        rng = np.random.default_rng(seed)
        result = data.copy()
        mask = rng.random(size=data.shape) < dropout_rate
        result[mask] = fill_value
        return result

    @staticmethod
    def with_bias(data: NDArray, bias: float = 0.5) -> NDArray:
        """
        Add constant offset/bias to data.

        Common in uncalibrated sensors (IMUs, force sensors, etc.).

        Parameters
        ----------
        data : NDArray
            Input data array
        bias : float, optional
            Constant bias to add (default 0.5)

        Returns
        -------
        NDArray
            Data with added bias

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> biased = corrupt.with_bias(data, bias=1.5)
        >>> np.allclose(biased - data, 1.5)
        True
        """
        return data + bias

    @staticmethod
    def with_drift(
        data: NDArray,
        drift_rate: float = 0.01,
        drift_type: str = "linear"
    ) -> NDArray:
        """
        Add time-dependent drift to data.

        Common in sensors that warm up or degrade (temperature sensors,
        gyroscopes, pressure sensors).

        Parameters
        ----------
        data : NDArray
            Input data array
        drift_rate : float, optional
            Rate of drift per sample (default 0.01)
        drift_type : str, optional
            Type of drift: 'linear' or 'exponential' (default 'linear')

        Returns
        -------
        NDArray
            Data with added drift

        Examples
        --------
        >>> import numpy as np
        >>> data = np.ones(5)
        >>> drifted = corrupt.with_drift(data, drift_rate=0.1)
        >>> drifted[-1] > drifted[0]  # drift increases over time
        True
        """
        n = len(data)
        if drift_type == "linear":
            drift = drift_rate * np.arange(n)
        elif drift_type == "exponential":
            drift = drift_rate * (np.exp(0.1 * np.arange(n)) - 1)
        else:
            raise ValueError(f"Unknown drift_type: {drift_type}")
        return data + drift

    @staticmethod
    def with_quantization(data: NDArray, levels: int = 256) -> NDArray:
        """
        Quantize data to discrete levels (ADC quantization).

        Simulates analog-to-digital conversion with limited bit depth.

        Parameters
        ----------
        data : NDArray
            Input data array
        levels : int, optional
            Number of quantization levels (default 256 for 8-bit ADC)

        Returns
        -------
        NDArray
            Quantized data

        Examples
        --------
        >>> import numpy as np
        >>> data = np.linspace(0, 1, 100)
        >>> quantized = corrupt.with_quantization(data, levels=10)
        >>> len(np.unique(quantized)) <= 10
        True
        """
        data_min, data_max = data.min(), data.max()
        if data_max == data_min:
            return data
        normalized = (data - data_min) / (data_max - data_min)
        quantized_norm = np.round(normalized * (levels - 1)) / (levels - 1)
        return data_min + quantized_norm * (data_max - data_min)

    @staticmethod
    def with_spikes(
        data: NDArray,
        spike_rate: float = 0.05,
        spike_magnitude: float = 5.0,
        seed: Optional[int] = None
    ) -> NDArray:
        """
        Add random spikes/outliers to data.

        Common in EMI, electrical interference, sensor glitches.

        Parameters
        ----------
        data : NDArray
            Input data array
        spike_rate : float, optional
            Fraction of data points to spike (default 0.05)
        spike_magnitude : float, optional
            Magnitude of spikes relative to data range (default 5.0)
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        NDArray
            Data with random spikes

        Examples
        --------
        >>> import numpy as np
        >>> data = np.ones(100)
        >>> spiked = corrupt.with_spikes(data, spike_rate=0.1, seed=42)
        >>> (np.abs(spiked - data) > 1).sum() > 0  # some spikes present
        True
        """
        rng = np.random.default_rng(seed)
        result = data.copy()
        mask = rng.random(size=data.shape) < spike_rate
        spikes = spike_magnitude * rng.choice([-1, 1], size=data.shape)
        result[mask] += spikes[mask]
        return result

    @staticmethod
    def with_clipping(
        data: NDArray,
        lower: Optional[float] = None,
        upper: Optional[float] = None
    ) -> NDArray:
        """
        Clip data to saturation limits (sensor saturation).

        Common when sensors reach their measurement range limits.

        Parameters
        ----------
        data : NDArray
            Input data array
        lower : float, optional
            Lower clipping bound (default: data.min())
        upper : float, optional
            Upper clipping bound (default: data.max())

        Returns
        -------
        NDArray
            Clipped data

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([-2., -1., 0., 1., 2.])
        >>> clipped = corrupt.with_clipping(data, lower=-1, upper=1)
        >>> np.allclose(clipped, [-1., -1., 0., 1., 1.])
        True
        """
        lower = lower if lower is not None else data.min()
        upper = upper if upper is not None else data.max()
        return np.clip(data, lower, upper)

    @staticmethod
    def with_delay(data: NDArray, delay: int = 1, fill_value: float = 0.0) -> NDArray:
        """
        Add time delay to data (latency, slow sensors).

        Common in communication delays, slow sensors, processing lag.

        Parameters
        ----------
        data : NDArray
            Input data array
        delay : int, optional
            Number of samples to delay (default 1)
        fill_value : float, optional
            Value to use for initial samples (default 0.0)

        Returns
        -------
        NDArray
            Delayed data

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1., 2., 3., 4.])
        >>> delayed = corrupt.with_delay(data, delay=2, fill_value=0)
        >>> np.allclose(delayed, [0., 0., 1., 2.])
        True
        """
        if delay <= 0:
            return data
        result = np.empty_like(data)
        result[:delay] = fill_value
        result[delay:] = data[:-delay]
        return result


class prepare:
    """
    Clean and prepare corrupted sensor data.

    All methods are static and take NumPy arrays as input, returning
    cleaned versions. Designed to handle common hardware issues before
    feeding data to estimators/controllers.
    """

    @staticmethod
    def with_moving_average(data: NDArray, window: int = 3) -> NDArray:
        """
        Apply moving average filter (denoise, smooth).

        Simple low-pass filter effective for Gaussian noise.

        Parameters
        ----------
        data : NDArray
            Input data array
        window : int, optional
            Size of moving average window (default 3)

        Returns
        -------
        NDArray
            Smoothed data

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1., 5., 2., 6., 3.])
        >>> smoothed = prepare.with_moving_average(data, window=3)
        >>> smoothed.shape == data.shape
        True
        >>> abs(smoothed[2] - 2.6667) < 0.001  # average of [1, 5, 2]
        True
        """
        if window < 1:
            return data
        result = np.empty_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.mean(data[start:i + 1])
        return result

    @staticmethod
    def with_median_filter(data: NDArray, window: int = 3) -> NDArray:
        """
        Apply median filter (remove spikes, outliers).

        Highly effective for spike/impulse noise while preserving edges.

        Parameters
        ----------
        data : NDArray
            Input data array
        window : int, optional
            Size of median window (default 3)

        Returns
        -------
        NDArray
            Filtered data

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1., 1., 100., 1., 1.])  # spike at index 2
        >>> filtered = prepare.with_median_filter(data, window=3)
        >>> filtered[2] < 10  # spike removed
        True
        """
        if window < 1:
            return data
        result = np.empty_like(data)
        half_window = window // 2
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            result[i] = np.median(data[start:end])
        return result

    @staticmethod
    def with_exponential_smoothing(data: NDArray, alpha: float = 0.3) -> NDArray:
        """
        Apply exponential smoothing filter (denoise, low-pass).

        Gives more weight to recent data. Alpha=1 is no filtering,
        alpha=0 is infinite smoothing.

        Parameters
        ----------
        data : NDArray
            Input data array
        alpha : float, optional
            Smoothing factor between 0 and 1 (default 0.3)

        Returns
        -------
        NDArray
            Smoothed data

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1., 2., 3., 4., 5.])
        >>> smoothed = prepare.with_exponential_smoothing(data, alpha=0.5)
        >>> smoothed.shape == data.shape
        True
        >>> smoothed[0] == data[0]  # first value unchanged
        True
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        result = np.empty_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def with_debounce(
        data: NDArray,
        threshold: float = 0.1,
        min_duration: int = 2
    ) -> NDArray:
        """
        Remove contact bounce from binary/step signals.

        Requires signal to remain stable for min_duration samples
        before accepting the transition.

        Parameters
        ----------
        data : NDArray
            Input data array
        threshold : float, optional
            Threshold for detecting state change (default 0.1)
        min_duration : int, optional
            Minimum stable samples required (default 2)

        Returns
        -------
        NDArray
            Debounced data

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([0., 0., 1., 0., 1., 1., 1.])  # bouncing
        >>> debounced = prepare.with_debounce(data, min_duration=2)
        >>> debounced.shape == data.shape
        True
        """
        result = data.copy()
        current_state = data[0]
        stable_count = 0

        for i in range(1, len(data)):
            if np.abs(data[i] - current_state) > threshold:
                stable_count += 1
                if stable_count >= min_duration:
                    current_state = data[i]
                    stable_count = 0
            else:
                stable_count = 0
            result[i] = current_state

        return result

    @staticmethod
    def with_outlier_removal(
        data: NDArray,
        threshold: float = 3.0,
        method: str = "replace"
    ) -> NDArray:
        """
        Detect and handle outliers using z-score method.

        Parameters
        ----------
        data : NDArray
            Input data array
        threshold : float, optional
            Z-score threshold for outlier detection (default 3.0)
        method : str, optional
            How to handle outliers: 'replace' with median or 'interpolate'
            (default 'replace')

        Returns
        -------
        NDArray
            Data with outliers handled

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1., 1., 100., 1., 1.])  # outlier at index 2
        >>> cleaned = prepare.with_outlier_removal(data, threshold=1.5)
        >>> cleaned[2] < 10  # outlier replaced
        True
        """
        result = data.copy()
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return result

        z_scores = np.abs((data - mean) / std)
        outliers = z_scores > threshold

        if method == "replace":
            median = np.median(data)
            result[outliers] = median
        elif method == "interpolate":
            # Linear interpolation between neighboring valid points
            valid = ~outliers
            if np.any(valid):
                valid_indices = np.where(valid)[0]
                result[outliers] = np.interp(
                    np.where(outliers)[0],
                    valid_indices,
                    data[valid]
                )
        else:
            raise ValueError(f"Unknown method: {method}")

        return result

    @staticmethod
    def with_interpolation(data: NDArray, method: str = "linear") -> NDArray:
        """
        Interpolate missing data (NaN values).

        Useful for handling dropouts and missing sensor readings.

        Parameters
        ----------
        data : NDArray
            Input data array (may contain NaN)
        method : str, optional
            Interpolation method: 'linear' or 'nearest' (default 'linear')

        Returns
        -------
        NDArray
            Data with NaN values interpolated

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1., np.nan, 3., np.nan, 5.])
        >>> filled = prepare.with_interpolation(data)
        >>> np.isnan(filled).sum() == 0  # no NaN remaining
        True
        """
        result = data.copy()
        nans = np.isnan(result)

        if not np.any(nans):
            return result

        valid = ~nans
        if not np.any(valid):
            # All NaN, return zeros
            return np.zeros_like(data)

        valid_indices = np.where(valid)[0]
        nan_indices = np.where(nans)[0]

        if method == "linear":
            result[nans] = np.interp(nan_indices, valid_indices, data[valid])
        elif method == "nearest":
            for i in nan_indices:
                nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - i))]
                result[i] = data[nearest_idx]
        else:
            raise ValueError(f"Unknown method: {method}")

        return result

    @staticmethod
    def with_staleness_policy(
        data: NDArray,
        policy: str = "hold"
    ) -> NDArray:
        """
        Apply staleness policy to data with missing values (NaN).

        Handles stale/missing sensor data according to different policies,
        matching the ROSNode staleness configuration in ros_node.py.

        This is particularly useful for sensor fusion where different
        sensors update at different rates, or when dealing with
        intermittent communication.

        Parameters
        ----------
        data : NDArray
            Input data array (may contain NaN for stale/missing data)
        policy : str, optional
            Staleness policy (default 'hold'):
            - 'zero': Replace missing/stale data with zeros
            - 'hold': Hold last valid value (forward fill)
            - 'drop': Remove data points with NaN (returns shorter array)
            - 'none': Keep NaN values as-is (no processing)

        Returns
        -------
        NDArray
            Processed data according to policy

        Examples
        --------
        Hold policy (forward fill - default):

        >>> import numpy as np
        >>> data = np.array([1., 2., np.nan, np.nan, 5.])
        >>> filled = prepare.with_staleness_policy(data, policy='hold')
        >>> np.array_equal(filled, [1., 2., 2., 2., 5.])
        True

        Zero policy (replace with zeros):

        >>> data = np.array([1., 2., np.nan, np.nan, 5.])
        >>> filled = prepare.with_staleness_policy(data, policy='zero')
        >>> np.array_equal(filled, [1., 2., 0., 0., 5.])
        True

        Drop policy (remove NaN entries):

        >>> data = np.array([1., 2., np.nan, np.nan, 5.])
        >>> filled = prepare.with_staleness_policy(data, policy='drop')
        >>> np.array_equal(filled, [1., 2., 5.])
        True

        None policy (keep NaN as-is):

        >>> data = np.array([1., 2., np.nan, np.nan, 5.])
        >>> filled = prepare.with_staleness_policy(data, policy='none')
        >>> np.array_equal(filled, data, equal_nan=True)
        True
        """
        if policy == "none":
            # Keep data as-is, including NaN values
            return data.copy()

        elif policy == "zero":
            # Replace NaN with zeros
            result = data.copy()
            result[np.isnan(result)] = 0.0
            return result

        elif policy == "hold":
            # Forward fill: hold last valid value
            result = data.copy()
            nans = np.isnan(result)

            if not np.any(nans):
                return result

            # Find first valid value
            valid_indices = np.where(~nans)[0]
            if len(valid_indices) == 0:
                # All NaN, return zeros
                return np.zeros_like(data)

            # Forward fill from first valid value
            last_valid = result[valid_indices[0]]
            for i in range(len(result)):
                if np.isnan(result[i]):
                    result[i] = last_valid
                else:
                    last_valid = result[i]

            return result

        elif policy == "drop":
            # Remove NaN entries entirely
            return data[~np.isnan(data)]

        else:
            raise ValueError(
                f"Unknown policy: {policy}. "
                f"Must be 'zero', 'hold', 'drop', or 'none'"
            )

    @staticmethod
    def with_calibration(data: NDArray, offset: float = 0.0, scale: float = 1.0) -> NDArray:
        """
        Apply calibration (remove bias, scale correction).

        Parameters
        ----------
        data : NDArray
            Input data array
        offset : float, optional
            Offset to subtract (bias correction) (default 0.0)
        scale : float, optional
            Scale factor to apply (default 1.0)

        Returns
        -------
        NDArray
            Calibrated data

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([2., 4., 6.])
        >>> calibrated = prepare.with_calibration(data, offset=1.0, scale=0.5)
        >>> np.allclose(calibrated, [0.5, 1.5, 2.5])
        True
        """
        return (data - offset) * scale

    @staticmethod
    def with_low_pass_filter(
        data: NDArray,
        alpha: float = 0.2
    ) -> NDArray:
        """
        Simple first-order low-pass filter (RC filter).

        Attenuates high-frequency noise while preserving low-frequency signals.

        Parameters
        ----------
        data : NDArray
            Input data array
        alpha : float, optional
            Filter coefficient (0=max filtering, 1=no filtering) (default 0.2)

        Returns
        -------
        NDArray
            Filtered data

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([0., 1., 0., 1., 0.])  # high freq
        >>> filtered = prepare.with_low_pass_filter(data, alpha=0.3)
        >>> filtered.shape == data.shape
        True
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        result = np.empty_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def with_clipping_recovery(
        data: NDArray,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        mark_invalid: bool = False
    ) -> NDArray:
        """
        Detect and handle clipped/saturated values.

        Parameters
        ----------
        data : NDArray
            Input data array
        lower : float, optional
            Lower saturation limit (default: data.min())
        upper : float, optional
            Upper saturation limit (default: data.max())
        mark_invalid : bool, optional
            If True, replace clipped values with NaN (default False)

        Returns
        -------
        NDArray
            Data with clipped values handled

        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1., 5., 5., 5., 1.])  # clipped at 5
        >>> recovered = prepare.with_clipping_recovery(data, upper=5, mark_invalid=True)
        >>> np.isnan(recovered[1:4]).all()  # clipped values marked
        True
        """
        result = data.copy()

        if lower is not None:
            lower_clip = np.abs(data - lower) < 1e-6
        else:
            lower_clip = np.zeros(len(data), dtype=bool)

        if upper is not None:
            upper_clip = np.abs(data - upper) < 1e-6
        else:
            upper_clip = np.zeros(len(data), dtype=bool)

        clipped = lower_clip | upper_clip

        if mark_invalid:
            result[clipped] = np.nan

        return result
