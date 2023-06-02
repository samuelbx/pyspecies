import warnings

import numpy as np
import numpy.random as npr
from tqdm import tqdm

from pyspecies._utils import XtoUV
from pyspecies.models import Model

FPS = 20
LENGTH = 7
TAU_LEAP = int(2e3)


class Discrete(Model):
    """Discrete model of population dynamics, simulated using Gillespie algorithm."""

    def __init__(self, D: np.ndarray, R: np.ndarray, avg_density: int) -> None:
        if avg_density < 50:
            warnings.warn(
                "Average density must be at least 100 for tau-leaping to work. Results might be inaccurate."
            )
        self.D, self.R, self.avg_density = D, R, avg_density

    def sim(self, X0_f: np.ndarray, Space: np.ndarray, Time: np.ndarray):
        """Simulates discrete steps of population dynamics using Gillespie algorithm."""
        N = Time.shape[0]
        X0 = np.zeros((2, len(X0_f) // 2))
        X0[0, :], X0[1, :] = XtoUV(X0_f)
        X0 = np.floor(self.avg_density * X0) / self.avg_density
        X = X0.copy()
        self.D *= len(Space) ** 2

        evol, Tlist = np.zeros((N * TAU_LEAP, 4), dtype=int), np.zeros(N * TAU_LEAP)
        for i in tqdm(range(N), "Simulating steps"):
            binom = npr.binomial(1, 0.5, TAU_LEAP)
            exp = npr.exponential(1, TAU_LEAP)

            rates, param = self.update_rates(X[0, :], X[1, :])
            events_sample = self.draw_events(rates)

            # Add random directions to events
            dirs = 2 * binom - 1
            events = np.zeros((len(events_sample[0]), 4), dtype=int)
            events[:, :3] = events_sample.T
            events[:, 3] = ((events[:, 2] + dirs) % len(Space)).astype(int)

            # Store events
            rng = i * TAU_LEAP + np.arange(TAU_LEAP)
            evol[rng] = events
            Tlist[rng] = np.cumsum(exp) / param

            # Update concentrations
            for event in events:
                self.process_event(X, *event)

        self.D /= len(Space) ** 2
        return self.rebuild_evolution(X0, evol, Tlist)

    def rebuild_evolution(
        self, X0: np.ndarray, evol: list[np.ndarray], Tlist: list[float], length=LENGTH
    ) -> list[np.ndarray]:
        """Rebuild the evolution of the system from the list of events, skipping unnecessary frames."""
        Xlist = [X0.copy()]
        nbr_frames = FPS * length
        Tlist_new = [0]
        X_current = X0.copy()
        for i, event in tqdm(enumerate(evol), "Rebuilding evolution"):
            self.process_event(X_current, *event)
            if i % np.ceil(len(evol) / nbr_frames) == 0:
                Xlist.append(X_current.flatten().copy())
                Tlist_new.append(Tlist_new[-1] + Tlist[i])
        Xlist[0] = Xlist[0].flatten()
        return Xlist, Tlist_new

    def update_rates(self, U: np.ndarray, V: np.ndarray):
        """Update the transition rates of possible events. The rates are stored in a 3D array of shape (3, 2, K) where K is the number of space points, and the first dimension corresponds to the event type (0 for diffusion, 1 for birth, 2 for death) and the second dimension corresponds to the species (0 for U, 1 for V)."""
        rates = np.zeros((3, 2, len(U)))
        rates[0, 0, :] = U * (self.D[0, 0] + self.D[0, 1] * U + self.D[0, 2] * V)
        rates[0, 1, :] = V * (self.D[1, 0] + self.D[1, 1] * U + self.D[1, 2] * V)
        rates[1, 0, :] = U * self.R[0, 0]
        rates[1, 1, :] = V * self.R[1, 0]
        rates[2, 0, :] = U * (self.R[0, 1] * U + self.R[0, 2] * V)
        rates[2, 1, :] = V * (self.R[1, 1] * U + self.R[1, 2] * V)
        return rates / rates.sum(), rates.sum()

    def draw_events(self, rates: np.ndarray) -> np.ndarray:
        """Draw events from the rates."""
        idx = npr.choice(np.arange(rates.size), p=rates.flatten(), size=TAU_LEAP)
        return np.stack(np.unravel_index(idx, rates.shape))

    def process_event(
        self, X: np.ndarray, event_type: int, species: int, start: int, end: int
    ):
        """Process an event (diffusion, birth or death) by updating the concentrations. End is always passed but only used for diffusion events."""
        if event_type == 0 and X[species, start] > 1 / self.avg_density:
            X[species, start] -= 1 / self.avg_density
            X[species, end] += 1 / self.avg_density

        elif event_type == 1:
            X[species, start] += 1 / self.avg_density

        elif event_type == 2:
            X[species, start] -= 1 / self.avg_density
            if X[species, start] < 0:
                X[species, start] = 0
