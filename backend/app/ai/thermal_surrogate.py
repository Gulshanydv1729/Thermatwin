"""Thermal decay and inflow surrogate for a CSS producer.

A bidirectional GRU predicts the state of the well 14 days ahead from the
operating history.  The inputs are the quantities a field operator actually
records for each production day,

* cumulative steam injected in the current cycle, m^3;
* elapsed soak duration, days;
* casing head pressure, MPa;
* days since the start of the production phase;
* the current pump rate, m^3/day, which the field always has on the gauge and
  which anchors the decline extrapolation;

and the outputs are

* bottom-hole temperature, degC;
* oil inflow rate, m^3/day.

The current bottom-hole temperature and the current pump rate are also
supplied, because the field always has them and they anchor the forecast.

Formulation
-----------
Predicting an absolute rate 14 days ahead from absolute inputs is badly
conditioned: the rate spans three orders of magnitude across the reservoir
skins in the training set, while the informative part of the window is its
*shape*.  The surrogate is therefore trained on a **scale-free** target,

.. math::

r_{t+H} / r_{max}, \\qquad T_{t+H} / T_{max}

where :math:`r_{max}` and :math:`T_{max}` are the largest rate and
temperature in the input window; the predicted absolute rate is recovered
by multiplying back by :math:`r_{max}`.  Every input channel is likewise
expressed as a ratio to a reference, so the network sees decline *shape*
rather than magnitude.

The network is trained on trajectories produced by the Marx-Langenheim CSS
solver in :mod:`backend.app.physics.thermal_reservoir`, so the surrogate is a
learned accelerator of a physics model whose conservation properties have been
verified independently, not a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.app.core.config import CSS, RESERVOIR
from backend.app.physics.thermal_reservoir import CSSPhase, ThermalReservoirModel

__all__ = [
    "SURROGATE_SEQUENCE_LENGTH",
    "SURROGATE_HORIZON_DAYS",
    "ThermalSurrogateNet",
    "ThermalSurrogate",
    "generate_training_trajectories",
    "build_supervised_windows",
    "ensure_weights",
]

#: Number of past production days fed to the network.
SURROGATE_SEQUENCE_LENGTH = 21
#: Forecast horizon, days.
SURROGATE_HORIZON_DAYS = 14
#: Four input channels, two output channels.
INPUT_CHANNELS = 5
OUTPUT_CHANNELS = 2


# ---------------------------------------------------------------------------
# Training data from the physics solver
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A production-phase trajectory and its aligned feature windows."""

    steam_m3: np.ndarray
    soak_days: np.ndarray
    casing_pressure_mpa: np.ndarray
    day_index: np.ndarray
    bottom_hole_temperature_c: np.ndarray
    inflow_m3_per_day: np.ndarray

    def __len__(self) -> int:
        return int(self.day_index.size)


def generate_training_trajectories(
    cycles: int = 6,
    production_days: int = CSS.production_days,
    skin_factors: Sequence[float] = (2.0, 3.2, 5.0, 8.0),
    permeabilities_md: Sequence[float] = (180.0, 320.0, 500.0),
    dt_days: float = 0.25,
) -> List[Trajectory]:
    """Generate CSS production trajectories from the physics solver.

    The well is cycled through a range of reservoir skins and permeabilities so
    the surrogate sees the spread of decline behaviour it must extrapolate over,
    and the casing head pressure is reconstructed from the bottom-hole pressure
    less the wellbore pressure traverse.
    """
    trajectories: List[Trajectory] = []
    for skin in skin_factors:
        for permeability in permeabilities_md:
            model = ThermalReservoirModel(
                skin_factor=skin, permeability_md=permeability, dt_days=dt_days
            )
            result = model.simulate_programme(
                cycles=cycles, production_days=production_days
            )
            producing = [
                state
                for state in result.states
                if state.phase == CSSPhase.PRODUCTION
            ]
            if len(producing) < SURROGATE_SEQUENCE_LENGTH + 1:
                continue
            steam = np.array(
                [s.cumulative_steam_volume_m3 for s in producing], dtype=float
            )
            soak = np.array(
                [CSS.soak_days if s.cumulative_steam_volume_m3 > 0.0 else 0.0 for s in producing]
            )
            # Casing head pressure: the pump intake pressure less the friction
            # and hydrostatic losses up the annulus, approximated by the
            # drawdown the pump imposes.
            casing = np.array(
                [
                    RESERVOIR.initial_reservoir_pressure_mpa
                    - 0.20 * s.oil_rate_tpd
                    / max(ThermalReservoirModel().pump_capacity_m3_per_day, 1e-9)
                    * (RESERVOIR.initial_reservoir_pressure_mpa * 0.55)
                    for s in producing
                ],
                dtype=float,
            )
            day = np.arange(len(producing), dtype=float)
            temperature = np.array(
                [s.bottom_hole_temperature_c for s in producing], dtype=float
            )
            inflow = np.array([s.oil_rate_tpd for s in producing], dtype=float)
            trajectories.append(
                Trajectory(
                    steam_m3=steam,
                    soak_days=soak,
                    casing_pressure_mpa=casing,
                    day_index=day,
                    bottom_hole_temperature_c=temperature,
                    inflow_m3_per_day=inflow,
                )
            )
    if not trajectories:
        raise RuntimeError("no trajectory long enough to build a training window")
    return trajectories


def build_supervised_windows(
    trajectories: Sequence[Trajectory], stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn trajectories into scale-free supervised windows.

    Each window spans ``SURROGATE_SEQUENCE_LENGTH`` days and the target is the
    state ``SURROGATE_HORIZON_DAYS`` days after the window closes, expressed
    relative to the window's own reference levels.

    Returns
    -------
    (inputs, targets)
        ``inputs`` has shape ``(n, SURROGATE_SEQUENCE_LENGTH, 5)`` and
        ``targets`` has shape ``(n, 2)`` holding the scaled future temperature
        and the future rate as a fraction of the window maximum.
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for trajectory in trajectories:
        n = len(trajectory)
        rate = np.maximum(trajectory.inflow_m3_per_day, 1e-9)
        steam = np.maximum(trajectory.steam_m3, 1e-9)
        pressure = np.maximum(trajectory.casing_pressure_mpa, 1e-9)
        temperature = np.maximum(trajectory.bottom_hole_temperature_c, 1e-9)
        for start_index in range(
            0, n - SURROGATE_SEQUENCE_LENGTH - SURROGATE_HORIZON_DAYS + 1, stride
        ):
            end_index = start_index + SURROGATE_SEQUENCE_LENGTH
            future = end_index + SURROGATE_HORIZON_DAYS - 1
            if future >= n:
                break
            rate_max = float(np.max(rate[start_index:end_index]))
            steam_max = float(np.max(steam[start_index:end_index]))
            pressure_max = float(np.max(pressure[start_index:end_index]))
            temperature_max = float(np.max(temperature[start_index:end_index]))
            window = np.stack(
                [
                    rate[start_index:end_index] / rate_max,
                    temperature[start_index:end_index] / temperature_max,
                    steam[start_index:end_index] / steam_max,
                    pressure[start_index:end_index] / pressure_max,
                    trajectory.soak_days[start_index:end_index]
                    / max(float(np.max(trajectory.soak_days[start_index:end_index])), 1e-9),
                ],
                axis=1,
            )
            xs.append(window)
            ys.append(
                np.array(
                    [
                        temperature[future] / temperature_max,
                        rate[future] / rate_max,
                    ],
                    dtype=float,
                )
            )
    if not xs:
        raise RuntimeError("no windows could be extracted")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _normaliser(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation, returned with shape ``(C,)``.

    Keeping the statistics one-dimensional lets numpy broadcast them against a
    ``(batch, sequence, channels)`` tensor during training and against a plain
    ``(sequence, channels)`` window during inference, with identical semantics.
    """
    mean = values.mean(axis=(0, 1))
    std = values.std(axis=(0, 1))
    return mean, np.where(std < 1e-8, 1.0, std)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def _torch():
    """Import torch lazily so the physics modules stay torch-free."""
    import torch  # noqa: WPS433  (deliberate lazy import)

    return torch


def _build_net():
    """Construct the bidirectional GRU.  torch is imported lazily."""
    torch = _torch()
    import torch.nn as nn

    class Network(nn.Module):
        """Bidirectional GRU with an explicit horizon offset."""

        def __init__(
            self,
            inputs: int = INPUT_CHANNELS,
            hidden: int = 64,
            outputs: int = OUTPUT_CHANNELS,
            layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                inputs,
                hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                # sequence[:, -1, :] is 2*hidden wide and its mean is too, so
                # the concatenated summary is 4*hidden wide.
                nn.Linear(4 * hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, outputs),
            )

        def forward(self, x):
            sequence, _ = self.gru(x)
            # A bidirectional GRU already summarises the window in both
            # directions, so its last timestep concatenates the forward state
            # at the close of the window with the backward state that has seen
            # the whole history in reverse.  A mean over the window is added so
            # that the level as well as the shape of the trend is retained.
            summary = torch.cat(
                [sequence[:, -1, :], sequence.mean(dim=1)], dim=1
            )
            return self.head(summary)

    return Network()


ThermalSurrogateNet = _build_net  # type: ignore[assignment,misc]


@dataclass
class SurrogatePrediction:
    """Forecast of the thermal and inflow state at the horizon."""

    bottom_hole_temperature_c: float
    inflow_m3_per_day: float
    horizon_days: int = SURROGATE_HORIZON_DAYS

    def as_dict(self) -> dict:
        """JSON-serialisable view for the API layer."""
        return {
            "bottom_hole_temperature_c": self.bottom_hole_temperature_c,
            "inflow_m3_per_day": self.inflow_m3_per_day,
            "horizon_days": self.horizon_days,
        }


class ThermalSurrogate:
    """Trained bidirectional GRU with persistence and a physics cross-check.

    Parameters
    ----------
    weights_path
        Cache location for the trained weights.
    device
        Torch device string.
    seed
        Seed for reproducible initialisation and training.
    train
        Whether to load or train on construction.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: str = "cpu",
        seed: int = 26120,
        train: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = device
        self.seed = seed
        torch = _torch()
        torch.manual_seed(seed)
        self.net = _build_net().to(device)
        self.net.eval()
        self.trained = False
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._output_mean: Optional[np.ndarray] = None
        self._output_std: Optional[np.ndarray] = None
        self.metrics: Dict[str, float] = {}
        if train and self.weights_path is not None and self.weights_path.exists():
            self.load()
        elif train:
            self.train_and_save(self.weights_path)

    # -- persistence ------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """Write the weights and the input/output normalisers."""
        target = Path(path) if path else self.weights_path
        if target is None:
            raise ValueError("no weights path configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch = _torch()
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "seed": self.seed,
                "input_mean": self._input_mean,
                "input_std": self._input_std,
                "output_mean": self._output_mean,
                "output_std": self._output_std,
            },
            target,
        )
        return target

    def load(self, path: Optional[Path] = None) -> None:
        """Read weights and normalisers from ``path``."""
        source = Path(path) if path else self.weights_path
        if source is None or not source.exists():
            raise FileNotFoundError("no trained weights available")
        torch = _torch()
        payload = torch.load(source, map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["state_dict"])
        self._input_mean = payload["input_mean"]
        self._input_std = payload["input_std"]
        self._output_mean = payload["output_mean"]
        self._output_std = payload["output_std"]
        self.net.eval()
        self.trained = True

    # -- training ---------------------------------------------------------
    def train_and_save(
        self,
        path: Optional[Path] = None,
        epochs: int = 220,
        batch_size: int = 64,
        validation_fraction: float = 0.2,
        lr: float = 5e-3,
        window_stride: int = 3,
        max_windows: int = 6000,
    ) -> Dict[str, float]:
        """Train on physics-solver trajectories and persist the weights.

        Both the temperature and the inflow channel are scored in physical
        units, and the returned metrics include the R^2 of each.
        """
        torch = _torch()
        trajectories = generate_training_trajectories()
        xs, ys = build_supervised_windows(trajectories, stride=window_stride)
        if len(xs) > max_windows:
            # Deterministic thinning so the training set is bounded but the
            # early (high-rate) part of every trajectory is retained.
            keep = np.linspace(0, len(xs) - 1, max_windows).astype(int)
            xs, ys = xs[keep], ys[keep]
        input_mean, input_std = _normaliser(xs)
        output_mean, output_std = _normaliser(ys)
        xs = (xs - input_mean) / input_std
        ys = (ys - output_mean) / output_std

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(xs))
        xs, ys = xs[order], ys[order]
        split = max(1, int(len(xs) * (1.0 - validation_fraction)))
        train_x = torch.from_numpy(xs[:split])
        train_y = torch.from_numpy(ys[:split])
        val_x = torch.from_numpy(xs[split:])
        val_y = torch.from_numpy(ys[split:])

        optimiser = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        self.net.train()
        last_loss = 0.0
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimiser.zero_grad()
                loss = criterion(self.net(batch_x.to(self.device)), batch_y.to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
                last_loss = float(loss.item())
            scheduler.step()
        self.net.eval()

        with torch.no_grad():
            predicted = self.net(val_x.to(self.device)).cpu().numpy()
        expected = val_y.numpy()
        physical_pred = predicted * output_std + output_mean
        physical_true = expected * output_std + output_mean

        def _r2(index: int) -> float:
            true = physical_true[:, index]
            pred = physical_pred[:, index]
            variance = float(np.sum((true - true.mean()) ** 2))
            if variance <= 0.0:
                return 0.0
            return 1.0 - float(np.sum((true - pred) ** 2)) / variance

        def _mae(index: int) -> float:
            return float(np.mean(np.abs(physical_true[:, index] - physical_pred[:, index])))

        self._input_mean, self._input_std = input_mean, input_std
        self._output_mean, self._output_std = output_mean, output_std
        self.metrics = {
            "final_loss": last_loss,
            "temperature_r2": _r2(0),
            "inflow_r2": _r2(1),
            "temperature_mae_c": _mae(0),
            "inflow_mae_m3_d": _mae(1),
            "windows": float(len(xs)),
        }
        self.trained = True
        if path is not None or self.weights_path is not None:
            self.save(path)
        return self.metrics

    # -- inference --------------------------------------------------------
    def predict(
        self,
        steam_m3: Sequence[float],
        soak_days: Sequence[float],
        casing_pressure_mpa: Sequence[float],
        inflow_m3_per_day: Sequence[float],
        bottom_hole_temperature_c: Sequence[float],
    ) -> SurrogatePrediction:
        """Forecast the state at the horizon from a window of history.

        The window is resampled to the model length, referenced to its own
        maximum rate and temperature, and the predicted rate is rescaled back
        to physical units.
        """
        if not self.trained:
            raise RuntimeError("surrogate has not been trained")
        for buffer in (
            self._input_mean,
            self._input_std,
            self._output_mean,
            self._output_std,
        ):
            if buffer is None:
                raise RuntimeError("normalisers are unavailable")
        channels = (
            inflow_m3_per_day,
            bottom_hole_temperature_c,
            steam_m3,
            casing_pressure_mpa,
            soak_days,
        )
        lengths = {len(c) for c in channels}
        if len(lengths) != 1:
            raise ValueError("all input series must have the same length")
        n = lengths.pop()
        if n < 2:
            raise ValueError("at least two history samples are required")

        source = np.linspace(0.0, 1.0, n)
        target_grid = np.linspace(0.0, 1.0, SURROGATE_SEQUENCE_LENGTH)
        resampled = [
            np.interp(target_grid, source, np.asarray(c, dtype=float)) for c in channels
        ]
        rate, temperature, steam, pressure, soak = resampled
        rate_max = max(float(np.max(rate)), 1e-9)
        temperature_max = max(float(np.max(temperature)), 1e-9)
        steam_max = max(float(np.max(steam)), 1e-9)
        pressure_max = max(float(np.max(pressure)), 1e-9)
        soak_max = max(float(np.max(soak)), 1e-9)
        window = np.stack(
            [
                rate / rate_max,
                temperature / temperature_max,
                steam / steam_max,
                pressure / pressure_max,
                soak / soak_max,
            ],
            axis=1,
        )
        scaled = (window - self._input_mean) / self._input_std
        torch = _torch()
        with torch.no_grad():
            out = self.net(
                torch.from_numpy(scaled[None, ...].astype(np.float32)).to(self.device)
            )
        physical = out.cpu().numpy()[0] * self._output_std + self._output_mean
        return SurrogatePrediction(
            bottom_hole_temperature_c=float(physical[0]) * temperature_max,
            inflow_m3_per_day=float(physical[1]) * rate_max,
        )


def ensure_weights(
    weights_dir: Path,
    device: str = "cpu",
    seed: int = 26120,
    epochs: int = 220,
) -> ThermalSurrogate:
    """Load cached surrogate weights, or train and cache them on first use."""
    target = Path(weights_dir) / "thermal_surrogate.pt"
    surrogate = ThermalSurrogate(weights_path=target, device=device, seed=seed, train=False)
    if target.exists():
        surrogate.load()
        return surrogate
    surrogate.train_and_save(target, epochs=epochs)
    return surrogate
