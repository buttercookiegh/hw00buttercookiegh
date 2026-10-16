"""Single-feature linear regression, trained using PyTorch SGD.

The train-only normalizations make optimization stable on Hydropower.csv.
The returned coefficients and predictions are always in ORIGINAL units.
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt


class LinearRegression(torch.nn.Module):
    """y = w0 + w1*x, trained with mini-batch SGD.

    Parameters
    ----------
    learning_rate : float
        Step size for torch.optim.SGD.
    epochs : int
        Number of passes over the training set.
    batch_size : int
        Number of samples per optimization step.
    random_state : int
        Seed for reproducible mini-batch shuffling.
    """

    def __init__(self, learning_rate=0.05, epochs=3000,
                 batch_size=32, random_state=42):
        super().__init__()
        if learning_rate <= 0 or epochs < 1 or batch_size < 1:
            raise ValueError("learning_rate, epochs and batch_size must be positive")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)

        self.w0 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.w1 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.history = {"w0": [], "w1": [], "loss": []}
        self.is_fitted = False

    @staticmethod
    def _column(values, label):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"{label} must contain finite numeric values")
        return torch.as_tensor(arr, dtype=torch.float64)

    def _physical_coefficients(self):
        slope = self.y_scale * self.w1 / self.x_scale
        intercept = self.y_mean + self.y_scale * self.w0 - slope * self.x_mean
        return float(intercept.detach()), float(slope.detach())

    def forward(self, x):
        """Compute AnnualProduction from unscaled BCR inputs, returning a tensor."""
        if not self.is_fitted:
            raise RuntimeError("Call fit before forward or predict")
        x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
        x_scaled = (x - self.x_mean) / self.x_scale
        y_scaled = self.w0 + self.w1 * x_scaled
        return self.y_mean + self.y_scale * y_scaled

    def fit(self, x_train, y_train, x_test, y_test):
        """Fit on TRAIN ONLY, then compute and print test R^2."""
        xt = self._column(x_train, 'x_train')
        yt = self._column(y_train, 'y_train')
        xv = self._column(x_test, 'x_test')
        yv = self._column(y_test, 'y_test')
        if xt.numel() != yt.numel() or xv.numel() != yv.numel():
            raise ValueError('Feature and target lengths must match')
        if xt.numel() < 2 or xv.numel() < 2:
            raise ValueError('Need at least two train and two test samples')

        # Only training rows determine the normalization constants.
        self.x_mean = xt.mean()
        self.x_scale = xt.std(unbiased=False).clamp_min(1e-12)
        self.y_mean = yt.mean()
        self.y_scale = yt.std(unbiased=False).clamp_min(1e-12)
        self.is_fitted = True
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        self.history = {"w0": [], "w1": [], "loss": []}
        data = torch.utils.data.TensorDataset(xt, yt)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        self.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                # Train with MSE in standardized target units for stable SGD.
                pred_scaled = (self.forward(xb) - self.y_mean) / self.y_scale
                true_scaled = (yb - self.y_mean) / self.y_scale
                loss_scaled = self.loss_fn(pred_scaled, true_scaled)
                loss_scaled.backward()
                self.optimizer.step()

            with torch.no_grad():
                pred_train = self.forward(xt)
                original_mse = self.loss_fn(pred_train, yt).item()
                b, m = self._physical_coefficients()
                self.history['w0'].append(b)
                self.history['w1'].append(m)
                self.history['loss'].append(original_mse)

        self.eval()
        with torch.no_grad():
            prediction = self.forward(xv)
            ss_res = float(torch.sum((yv - prediction)**2))
            ss_tot = float(torch.sum((yv - yv.mean())**2))
        self.r2_test_ = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        self.intercept_, self.coef_ = self._physical_coefficients()
        print(f"Fitted model: AnnualProduction = {self.intercept_:.4f} + "
              f"({self.coef_:.4f}) * BCR")
        print(f"Test R²: {self.r2_test_:.6f}")
        print(f"Final training MSE: {self.history['loss'][-1]:.4f}")
        return self

    def predict(self, new_bcr):
        """Return AnnualProduction predictions as a NumPy array."""
        with torch.no_grad():
            return self.forward(self._column(new_bcr, 'new_bcr')).numpy()

    def analysis_plot(self, x_train, y_train, x_test=None, y_test=None):
        """Plot raw data and line, plus w0, w1 and loss trajectories."""
        if not self.is_fitted:
            raise RuntimeError('Call fit before analysis_plot')
        xt = np.asarray(x_train, dtype=float).reshape(-1)
        yt = np.asarray(y_train, dtype=float).reshape(-1)
        grid = np.linspace(float(xt.min()), float(xt.max()), 200)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        ax = axes[0, 0]
        ax.scatter(xt, yt, alpha=.65, s=20, label='Training observations')
        if x_test is not None and y_test is not None:
            ax.scatter(x_test, y_test, alpha=.65, s=20, label='Test observations')
        ax.plot(grid, self.predict(grid), linewidth=2, label='SGD-fitted line')
        ax.set(xlabel='BCR', ylabel='AnnualProduction', title='Original data and fitted regression')
        ax.legend()
        axes[0, 1].plot(self.history['w0'])
        axes[0, 1].set(xlabel='Epoch', ylabel='Original-unit intercept', title='w0 across epochs')
        axes[1, 0].plot(self.history['w1'])
        axes[1, 0].set(xlabel='Epoch', ylabel='Original-unit slope', title='w1 across epochs')
        axes[1, 1].plot(self.history['loss'])
        axes[1, 1].set(xlabel='Epoch', ylabel='Original-unit training MSE', title='Training loss')
        for axis in axes.flat:
            axis.grid(alpha=.2)
        plt.show()
        return fig
