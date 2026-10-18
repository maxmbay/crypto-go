"""Forecast designs that all see the same 48-hour market window.

Nothing in here reads community data. LightGBM is the tabular baseline.
The other designs are sequence models that may use the path, not just a
fixed lag summary.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

LOOKBACK = 48
N_FEATURES = 6
STATIC_DIM = 4


def prediction_head(in_dim: int, static_dim: int = STATIC_DIM) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim + static_dim, 64),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1),
    )


class MLP(nn.Module):
    def __init__(self, lookback: int = LOOKBACK, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback * n_features + static_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([x.flatten(1), static], dim=-1)
        return self.net(flat).squeeze(-1)


class NLinear(nn.Module):
    """Subtract the last value, map the window with one linear layer, add it back."""

    def __init__(self, lookback: int = LOOKBACK, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM):
        super().__init__()
        self.linear = nn.Linear(lookback, 1)
        self.head = prediction_head(n_features * 2, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        shifted = (x - last.unsqueeze(1)).transpose(1, 2)
        mapped = self.linear(shifted).squeeze(-1)
        return self.head(torch.cat([mapped, last, static], dim=-1)).squeeze(-1)


class DLinear(nn.Module):
    """Separate linear maps for a moving-average trend and the residual."""

    def __init__(self, lookback: int = LOOKBACK, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM):
        super().__init__()
        self.trend = nn.Linear(lookback, 1)
        self.seasonal = nn.Linear(lookback, 1)
        self.head = prediction_head(n_features, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        series = x.transpose(1, 2)
        trend = nn.functional.avg_pool1d(series, kernel_size=13, stride=1, padding=6)
        seasonal = series - trend
        mapped = (self.trend(trend) + self.seasonal(seasonal)).squeeze(-1)
        return self.head(torch.cat([mapped, static], dim=-1)).squeeze(-1)


class NBeats(nn.Module):
    def __init__(self, lookback: int = LOOKBACK, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM):
        super().__init__()
        self.n_features = n_features
        self.fc1 = nn.Sequential(nn.Linear(lookback, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.back1 = nn.Linear(64, lookback)
        self.fore1 = nn.Linear(64, 1)
        self.fc2 = nn.Sequential(nn.Linear(lookback, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.back2 = nn.Linear(64, lookback)
        self.fore2 = nn.Linear(64, 1)
        self.head = prediction_head(1, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        series = x.transpose(1, 2).reshape(-1, x.shape[1])
        hidden = self.fc1(series)
        residual = series - self.back1(hidden)
        forecast = self.fore1(hidden) + self.fore2(self.fc2(residual))
        forecast = forecast.view(x.shape[0], self.n_features).sum(dim=-1, keepdim=True)
        return self.head(torch.cat([forecast, static], dim=-1)).squeeze(-1)


class TSMixer(nn.Module):
    def __init__(self, lookback: int = LOOKBACK, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM):
        super().__init__()
        self.time_norm = nn.LayerNorm(lookback)
        self.time_mix = nn.Linear(lookback, lookback)
        self.feat_norm = nn.LayerNorm(n_features)
        self.feat_mix = nn.Linear(n_features, n_features)
        self.head = prediction_head(n_features, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        mixed = x + self.time_mix(self.time_norm(x.transpose(1, 2))).transpose(1, 2)
        mixed = mixed + self.feat_mix(self.feat_norm(mixed))
        return self.head(torch.cat([mixed[:, -1, :], static], dim=-1)).squeeze(-1)


class TCN(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM):
        super().__init__()
        channels = [n_features, 32, 32, 32, 32]
        dilations = [1, 2, 4, 8]
        blocks = []
        for i, dilation in enumerate(dilations):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, dilation=dilation, padding=dilation),
                    nn.GELU(),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = prediction_head(32, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        hidden = x.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(torch.cat([hidden[:, :, -1], static], dim=-1)).squeeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM, hidden: int = 64):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = prediction_head(hidden, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _out, (state, _cell) = self.rnn(x)
        return self.head(torch.cat([state[-1], static], dim=-1)).squeeze(-1)


class GRUForecaster(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM, hidden: int = 64):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        self.head = prediction_head(hidden, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _out, state = self.rnn(x)
        return self.head(torch.cat([state[-1], static], dim=-1)).squeeze(-1)


class _Encoder(nn.Module):
    def __init__(self, d_model: int, layers: int, nhead: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    def forward(
        self,
        hidden: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)


class TransformerForecaster(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.encoder = _Encoder(d_model, layers=2)
        self.head = prediction_head(d_model, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.proj(x))
        return self.head(torch.cat([hidden[:, -1, :], static], dim=-1)).squeeze(-1)


class TransformerGRU(nn.Module):
    """Self-attention over the hour window, then a GRU over that sequence."""

    def __init__(self, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.encoder = _Encoder(d_model, layers=1)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.head = prediction_head(d_model, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.proj(x))
        _out, state = self.rnn(hidden)
        return self.head(torch.cat([state[-1], static], dim=-1)).squeeze(-1)


class GRUTransformer(nn.Module):
    """GRU over the hours, then self-attention over the GRU outputs."""

    def __init__(self, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM, d_model: int = 64):
        super().__init__()
        self.rnn = nn.GRU(n_features, d_model, batch_first=True)
        self.encoder = _Encoder(d_model, layers=1)
        self.head = prediction_head(d_model, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        sequence, _state = self.rnn(x)
        hidden = self.encoder(sequence)
        return self.head(torch.cat([hidden[:, -1, :], static], dim=-1)).squeeze(-1)


class PatchTST(nn.Module):
    def __init__(
        self,
        lookback: int = LOOKBACK,
        n_features: int = N_FEATURES,
        static_dim: int = STATIC_DIM,
        patch: int = 4,
        d_model: int = 64,
    ):
        super().__init__()
        if lookback % patch != 0:
            raise ValueError("lookback must be divisible by patch length")
        self.patch = patch
        self.proj = nn.Linear(patch * n_features, d_model)
        self.encoder = _Encoder(d_model, layers=2)
        self.head = prediction_head(d_model, static_dim)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        batch, length, n_features = x.shape
        patched = x.reshape(batch, length // self.patch, self.patch * n_features)
        hidden = self.encoder(self.proj(patched)).mean(dim=1)
        return self.head(torch.cat([hidden, static], dim=-1)).squeeze(-1)


class CrossAssetEncoder(nn.Module):
    """One token per asset: a GRU summary of that asset, then attention across assets."""

    def __init__(self, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM, d_model: int = 64):
        super().__init__()
        self.rnn = nn.GRU(n_features, d_model, batch_first=True)
        self.proj = nn.Linear(d_model + static_dim, d_model)
        self.encoder = _Encoder(d_model, layers=2)

    def forward(self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        batch, n_assets, length, _n_features = x.shape
        _sequence, state = self.rnn(x.reshape(batch * n_assets, length, -1))
        tokens = state[-1].reshape(batch, n_assets, -1)
        return self.encoder(self.proj(torch.cat([tokens, static], dim=-1)), src_key_padding_mask=pad)


class CrossAssetTransformer(nn.Module):
    """Cross-asset tokens with a point head for the 24-hour log return."""

    def __init__(self, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM, d_model: int = 64):
        super().__init__()
        self.trunk = CrossAssetEncoder(n_features, static_dim, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x, static, pad)).squeeze(-1)


def _normal_quadrature(points: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights for E[g(Z)], Z standard normal."""
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    z = torch.tensor(nodes * np.sqrt(2.0), dtype=torch.float32)
    w = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float32)
    return z, w


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    edges = bin_locations.clone()
    edges[..., -1] = edges[..., -1] + 1e-6
    return torch.sum(inputs.unsqueeze(-1) >= edges, dim=-1) - 1


def _rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool,
    left: float,
    right: float,
    bottom: float,
    top: float,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monotonic rational-quadratic spline on a bounded interval. Durkan et al."""
    num_bins = unnormalized_widths.shape[-1]
    widths = torch.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = torch.nn.functional.pad(cumwidths, (1, 0), value=0.0)
    span = right - left
    cumwidths = span * cumwidths + left
    cumwidths = torch.cat([cumwidths[..., :1] * 0 + left, cumwidths[..., 1:-1], cumwidths[..., -1:] * 0 + right], dim=-1)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + torch.nn.functional.softplus(unnormalized_derivatives)
    heights = torch.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = torch.nn.functional.pad(cumheights, (1, 0), value=0.0)
    vspan = top - bottom
    cumheights = vspan * cumheights + bottom
    cumheights = torch.cat(
        [cumheights[..., :1] * 0 + bottom, cumheights[..., 1:-1], cumheights[..., -1:] * 0 + top],
        dim=-1,
    )
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    delta = heights / widths

    bin_idx = _searchsorted(cumheights if inverse else cumwidths, inputs).unsqueeze(-1)
    input_cumwidths = cumwidths.gather(-1, bin_idx).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx).squeeze(-1)
    input_cumheights = cumheights.gather(-1, bin_idx).squeeze(-1)
    input_heights = heights.gather(-1, bin_idx).squeeze(-1)
    input_delta = delta.gather(-1, bin_idx).squeeze(-1)
    input_derivatives = derivatives.gather(-1, bin_idx).squeeze(-1)
    input_derivatives_next = derivatives[..., 1:].gather(-1, bin_idx).squeeze(-1)

    if inverse:
        shift = inputs - input_cumheights
        gap = input_derivatives + input_derivatives_next - 2 * input_delta
        a = shift * gap + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - shift * gap
        c = -input_delta * shift
        discriminant = torch.clamp(b.square() - 4 * a * c, min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant) + 1e-8)
        theta = root
        outputs = root * input_bin_widths + input_cumwidths
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus = theta * (1 - theta)
        numerator = input_heights * (input_delta * theta.square() + input_derivatives * theta_one_minus)
        denominator = input_delta + (input_derivatives + input_derivatives_next - 2 * input_delta) * theta_one_minus
        outputs = input_cumheights + numerator / denominator

    theta_one_minus = theta * (1 - theta)
    denominator = input_delta + (input_derivatives + input_derivatives_next - 2 * input_delta) * theta_one_minus
    derivative_numerator = input_delta.square() * (
        input_derivatives_next * theta.square()
        + 2 * input_delta * theta_one_minus
        + input_derivatives * (1 - theta).square()
    )
    logabsdet = torch.log(derivative_numerator.clamp_min(1e-12)) - 2 * torch.log(denominator.clamp_min(1e-12))
    if inverse:
        logabsdet = -logabsdet
    return outputs, logabsdet


def _unconstrained_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool,
    data_bound: float,
    noise_bound: float,
    tail_slope: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spline from returns in [-data_bound, data_bound] onto noise in [-noise_bound, noise_bound].

    Outside that, a linear tail with slope dy/dz = `tail_slope`. The data bound
    has to sit near the returns we actually see. A shared bound of 2 puts a few
    percent of the base gaussian at a 200% move, and that dominates the variance.
    """
    if tail_slope <= 0 or data_bound <= 0 or noise_bound <= 0:
        raise ValueError("bounds and tail slope must be positive")
    min_derivative = 1e-3
    # Forward map is data -> noise, so the boundary derivative on that map is 1/(dy/dz).
    noise_slope = 1.0 / tail_slope
    constant = math.log(math.exp(noise_slope - min_derivative) - 1.0)
    derivatives = torch.nn.functional.pad(unnormalized_derivatives, (1, 1), value=constant)
    sign = torch.where(inputs >= 0, torch.ones_like(inputs), -torch.ones_like(inputs))
    if inverse:
        inside = inputs.abs() <= noise_bound
        tail_value = sign * data_bound + (inputs - sign * noise_bound) * tail_slope
        tail_log = torch.full_like(inputs, math.log(tail_slope))
    else:
        inside = inputs.abs() <= data_bound
        tail_value = sign * noise_bound + (inputs - sign * data_bound) / tail_slope
        tail_log = torch.full_like(inputs, -math.log(tail_slope))
    outside = ~inside
    outputs = torch.where(outside, tail_value, inputs)
    logabsdet = torch.where(outside, tail_log, torch.zeros_like(inputs))
    if not bool(inside.any()):
        return outputs, logabsdet
    transformed, det = _rational_quadratic_spline(
        inputs[inside],
        unnormalized_widths[inside],
        unnormalized_heights[inside],
        derivatives[inside],
        inverse=inverse,
        left=-data_bound,
        right=data_bound,
        bottom=-noise_bound,
        top=noise_bound,
        min_derivative=min_derivative,
    )
    outputs = outputs.index_put((inside,), transformed)
    logabsdet = logabsdet.index_put((inside,), det)
    return outputs, logabsdet


class ConditionalReturnFlow(nn.Module):
    """Conditional 1D neural spline flow. Context in, density over a scalar return out."""

    def __init__(
        self,
        context_dim: int,
        bins: int = 8,
        data_bound: float = 0.30,
        noise_bound: float = 3.0,
        tail_slope: float = 0.05,
        bound: float | None = None,
    ):
        super().__init__()
        self.bins = bins
        self.data_bound = bound if bound is not None else data_bound
        self.noise_bound = noise_bound
        self.tail_slope = tail_slope
        self.net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3 * bins - 1),
        )
        nodes, weights = _normal_quadrature(16)
        self.register_buffer("quad_z", nodes)
        self.register_buffer("quad_w", weights)

    def _spline_params(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(context)
        b = self.bins
        return raw[..., :b], raw[..., b : 2 * b], raw[..., 2 * b :]

    def _transform(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor,
        inverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        widths, heights, derivs = self._spline_params(context)
        flat_in = inputs.reshape(-1)
        flat_w = widths.reshape(-1, self.bins)
        flat_h = heights.reshape(-1, self.bins)
        flat_d = derivs.reshape(-1, self.bins - 1)
        outputs, logdet = _unconstrained_spline(
            flat_in,
            flat_w,
            flat_h,
            flat_d,
            inverse=inverse,
            data_bound=self.data_bound,
            noise_bound=self.noise_bound,
            tail_slope=self.tail_slope,
        )
        return outputs.reshape(inputs.shape), logdet.reshape(inputs.shape)

    def log_prob(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z, logdet = self._transform(y, context, inverse=False)
        log_base = -0.5 * (math.log(2.0 * math.pi) + z.square())
        return log_base + logdet

    def quantile(self, context: torch.Tensor, q: float) -> torch.Tensor:
        if not 0.0 < q < 1.0:
            raise ValueError("quantile must be strictly inside (0, 1)")
        z_value = math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * q - 1.0)))
        z = torch.full(context.shape[:-1], z_value, dtype=context.dtype, device=context.device)
        y, _logdet = self._transform(z, context, inverse=True)
        return y

    def median(self, context: torch.Tensor) -> torch.Tensor:
        return self.quantile(context, 0.5)

    def moments(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        samples = self.quad_z.shape[0]
        lead = context.shape[:-1]
        z = self.quad_z.view(samples, *([1] * len(lead))).expand(samples, *lead)
        ctx = context.unsqueeze(0).expand(samples, *lead, context.shape[-1])
        y, _logdet = self._transform(z, ctx, inverse=True)
        weight = self.quad_w.view(samples, *([1] * len(lead)))
        mean = (weight * y).sum(dim=0)
        var = (weight * y.square()).sum(dim=0) - mean.square()
        return mean, var.clamp_min(1e-12)

    def mean(self, context: torch.Tensor) -> torch.Tensor:
        return self.moments(context)[0]

    def prob_above(self, level: float, context: torch.Tensor) -> torch.Tensor:
        levels = torch.full(context.shape[:-1], float(level), dtype=context.dtype, device=context.device)
        z, _logdet = self._transform(levels, context, inverse=False)
        return 0.5 * torch.erfc(z / math.sqrt(2.0))


class CrossAssetFlow(nn.Module):
    """The cross-asset token conditions a spline density over the 24-hour log return."""

    def __init__(self, n_features: int = N_FEATURES, static_dim: int = STATIC_DIM, d_model: int = 64):
        super().__init__()
        self.trunk = CrossAssetEncoder(n_features, static_dim, d_model)
        self.density = ConditionalReturnFlow(d_model)

    def log_prob(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        pad: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.density.log_prob(y, self.trunk(x, static, pad))

    def mean(self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.density.mean(self.trunk(x, static, pad))

    def moments(
        self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.density.moments(self.trunk(x, static, pad))

    def median(self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.density.median(self.trunk(x, static, pad))

    def quantile(self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor, q: float) -> torch.Tensor:
        return self.density.quantile(self.trunk(x, static, pad), q)

    def prob_above(
        self,
        level: float,
        x: torch.Tensor,
        static: torch.Tensor,
        pad: torch.Tensor,
    ) -> torch.Tensor:
        return self.density.prob_above(level, self.trunk(x, static, pad))


SEQUENCE_MODELS: dict[str, type[nn.Module]] = {
    "mlp": MLP,
    "nlinear": NLinear,
    "dlinear": DLinear,
    "nbeats": NBeats,
    "tsmixer": TSMixer,
    "tcn": TCN,
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "transformer": TransformerForecaster,
    "transformer_gru": TransformerGRU,
    "gru_transformer": GRUTransformer,
    "patchtst": PatchTST,
}
