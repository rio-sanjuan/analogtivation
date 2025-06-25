"""Visual tests for activation function shapes."""

from time import struct_time
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest


class TestActivationVisualizations:
    """Test activation function shapes and visual properties."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create temporary directory for plots."""
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir(exist_ok=True)
        return plot_dir

    def test_plot_all_activations(self, output_dir):
        """Create plots for all activation functions."""
        from analogtivation.core import (
            fourier_activation,
            logistic_map_activation,
            lorenz_activation,
            mandelbrot_activation,
            quantum_activation,
            wave_activation,
        )

        # Test range
        x = np.linspace(-3, 3, 1000)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Plot each activation
        activations = [
            ("Wave", lambda x: wave_activation(x)),
            ("Fourier", lambda x: fourier_activation(x, n_terms=5)),
            ("Quantum", lambda x: quantum_activation(x, n_states=4)),
            ("Lorenz", lambda x: lorenz_activation(x)),
            ("Logistic Map", lambda x: logistic_map_activation(x, r=3.9)),
            ("Mandelbrot", lambda x: mandelbrot_activation(x, max_iter=5)),
        ]

        for idx, (name, activation) in enumerate(activations):
            y = activation(x)

            ax = axes[idx]
            ax.plot(x, y, "b-", linewidth=2)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{name} Activation", fontsize=12)
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")

            # Add zero lines
            ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "all_activations.png", dpi=150)
        plt.close()

        # Verify plot was created
        assert (output_dir / "all_activations.png").exists()

    @patch("analogtivation.core.time_based.gmtime")
    def test_time_based_activations_over_time(self, mock_gmtime, output_dir):
        """Plot time-based activations at different times."""
        from analogtivation.core import circadian_activation, clock_activation

        x = np.linspace(-2, 2, 100)

        # Test at different times
        times = [
            (0, "12:00 AM"),
            (6, "6:00 AM"),
            (12, "12:00 PM"),
            (18, "6:00 PM"),
        ]

        # Clock activation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for hour, label in times:
            mock_gmtime.return_value = struct_time((2024, 1, 1, hour, 0, 0, 0, 1, 0))
            y = clock_activation(x)
            ax1.plot(x, y, label=label, linewidth=2)

        ax1.set_title("Clock Activation at Different Times")
        ax1.set_xlabel("Input")
        ax1.set_ylabel("Output")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Circadian activation
        hours = np.arange(24)
        x_fixed = np.array([1.0])  # Fixed input
        circadian_values = []

        for hour in hours:
            mock_gmtime.return_value = struct_time((2024, 1, 1, hour, 0, 0, 0, 1, 0))
            circadian_values.append(circadian_activation(x_fixed)[0])

        ax2.plot(hours, circadian_values, "b-", linewidth=2)
        ax2.set_title("Circadian Activation Over 24 Hours (input=1.0)")
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Output")
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 25, 6))

        plt.tight_layout()
        plt.savefig(output_dir / "time_based_activations.png", dpi=150)
        plt.close()

        assert (output_dir / "time_based_activations.png").exists()

    def test_wave_activation_frequencies(self, output_dir):
        """Plot wave activation with different frequencies."""
        from analogtivation.core import wave_activation

        x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

        fig, ax = plt.subplots(figsize=(10, 6))

        frequencies = [0.5, 1.0, 2.0, 3.0]
        for freq in frequencies:
            y = wave_activation(x, frequencies=[freq], amplitudes=[1.0])
            ax.plot(x, y, label=f"freq={freq}", linewidth=2)

        ax.set_title("Wave Activation with Different Frequencies")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2 * np.pi, 2 * np.pi)

        plt.tight_layout()
        plt.savefig(output_dir / "wave_frequencies.png", dpi=150)
        plt.close()

        assert (output_dir / "wave_frequencies.png").exists()

    def test_chaos_activation_parameters(self, output_dir):
        """Plot chaos activations with different parameters."""
        from analogtivation.core import logistic_map_activation

        x = np.linspace(-3, 3, 500)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Different r values for logistic map
        r_values = [2.5, 3.2, 3.7, 3.9]

        for idx, r in enumerate(r_values):
            ax = axes[idx // 2, idx % 2]

            for iterations in [1, 3, 5]:
                y = logistic_map_activation(x, r=r, iterations=iterations)
                ax.plot(x, y, label=f"iter={iterations}", linewidth=2)

            ax.set_title(f"Logistic Map (r={r})")
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.1, 1.1)

        plt.tight_layout()
        plt.savefig(output_dir / "chaos_parameters.png", dpi=150)
        plt.close()

        assert (output_dir / "chaos_parameters.png").exists()

    def test_activation_derivatives(self, output_dir):
        """Plot activation functions and their derivatives."""
        from analogtivation.core import (
            FourierActivation,
            LorenzActivation,
            WaveActivation,
        )

        x = np.linspace(-3, 3, 1000)

        fig, axes = plt.subplots(3, 2, figsize=(10, 12))

        activations = [
            ("Wave", WaveActivation()),
            ("Fourier", FourierActivation(n_terms=5)),
            ("Lorenz", LorenzActivation()),
        ]

        for idx, (name, activation) in enumerate(activations):
            # Function values
            y = activation.forward(x)
            axes[idx, 0].plot(x, y, "b-", linewidth=2)
            axes[idx, 0].set_title(f"{name} Activation")
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].set_xlabel("Input")
            axes[idx, 0].set_ylabel("Output")

            # Derivatives
            dy = activation.gradient(x)
            axes[idx, 1].plot(x, dy, "r-", linewidth=2)
            axes[idx, 1].set_title(f"{name} Gradient")
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].set_xlabel("Input")
            axes[idx, 1].set_ylabel("Gradient")

        plt.tight_layout()
        plt.savefig(output_dir / "activation_derivatives.png", dpi=150)
        plt.close()

        assert (output_dir / "activation_derivatives.png").exists()

    def test_activation_heatmaps(self, output_dir):
        """Create 2D heatmaps showing activation behavior."""
        from analogtivation.core import mandelbrot_activation, quantum_activation

        # Create 2D grid
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Quantum activation on complex plane
        Z1 = quantum_activation(X + 1j * Y, n_states=4)
        im1 = ax1.imshow(
            np.real(Z1), extent=[-2, 2, -2, 2], origin="lower", cmap="viridis"
        )
        ax1.set_title("Quantum Activation (Real Part)")
        ax1.set_xlabel("Real")
        ax1.set_ylabel("Imaginary")
        plt.colorbar(im1, ax=ax1)

        # Mandelbrot activation
        Z2 = mandelbrot_activation(X, max_iter=10)
        im2 = ax2.imshow(Z2, extent=[-2, 2, -2, 2], origin="lower", cmap="hot")
        ax2.set_title("Mandelbrot Activation")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.savefig(output_dir / "activation_heatmaps.png", dpi=150)
        plt.close()

        assert (output_dir / "activation_heatmaps.png").exists()


class TestActivationProperties:
    """Test visual properties of activations."""

    def test_activation_smoothness(self):
        """Test that activations are visually smooth."""
        from analogtivation.core import fourier_activation, wave_activation

        x = np.linspace(-5, 5, 1000)

        # Wave activation should be smooth
        y_wave = wave_activation(x)
        differences = np.diff(y_wave)

        # Check smoothness (no large jumps)
        assert np.all(np.abs(differences) < 0.1)

        # Fourier should also be smooth
        y_fourier = fourier_activation(x, n_terms=10)
        differences_fourier = np.diff(y_fourier)
        assert np.all(np.abs(differences_fourier) < 0.5)

    def test_activation_symmetry(self):
        """Test symmetry properties visually."""
        from analogtivation.core import wave_activation

        x = np.linspace(-3, 3, 1000)

        # Wave with specific setup should have symmetry
        y = wave_activation(x, frequencies=[1.0], amplitudes=[1.0])

        # Check approximate odd symmetry for sine-based activation
        mid = len(x) // 2
        left = y[:mid]
        right = y[mid:][::-1]

        # Should be approximately odd symmetric
        # (not exact due to discretization)
        correlation = np.corrcoef(left, -right)[0, 1]
        assert correlation > 0.99
