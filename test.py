"""
This script is used to run tests
"""

import numpy as np

if __name__ == "__main__":
    # ── Initial conditions (n_sims,) ──────────────────────────────────────
    seed = 42
    n_sims = 3
    TARGET = 42.0
    dt = 0.1

    rng = np.random.default_rng(seed)

    speed     = rng.uniform(2.5, 4.5, size=n_sims)      # m/s
    fatigue_k = rng.uniform(0.0, 0.00002, size=n_sims)

    # ── State arrays ──────────────────────────────────────────────────────
    position  = np.zeros(n_sims)
    time      = np.zeros(n_sims)
    active    = np.ones(n_sims, dtype=bool)   # True = still running

    # ── Output arrays (pre-allocate worst case) ──────────────────────────
    finish_time = np.full(n_sims, np.nan)

    # for step in range(100):



        # if not active.any():
        #     break                             # all sims done → early exit

    # ── Per-step calculations (only on active sims) ───────────────────
    noise         = rng.normal(0.0, 0.03, size=n_sims)
    current_speed = np.where(active, speed * (1 - fatigue_k * time) + noise, 0.0)
    # current_speed = np.clip(current_speed, 0.1, None)

        # # ── Advance state ─────────────────────────────────────────────────
        # position += np.where(active, current_speed * dt, 0.0)
        # time     += np.where(active, dt, 0.0)

        # # ── Check finishing condition ─────────────────────────────────────
        # just_finished        = active & (position >= TARGET)
        # finish_time[just_finished] = time[just_finished]

        # # ── Deactivate finished sims ──────────────────────────────────────
        # active[just_finished] = False

    print(position)
    print(finish_time)
    print(noise)
    print(current_speed)