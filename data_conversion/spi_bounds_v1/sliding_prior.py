"""
Analytischer, voll differenzierbarer SlidingPrior für pilot_doe_v0.

- inputs_world x = [v_slide (m/s), F_N_mag (N)]  (static_input_size = 2)
- point_start s_in = [px, py, pz, qw, qx, qy, qz] im sliding_frame (m)
- Ausgabe: [B, T, 15] = [p_EOS, p_succ, px, py, pz, qw, qx, qy, qz,
                         Fx, Fy, Fz, Mx, My, Mz]
- Raster dt = 0,032 s, T = 640 standardmäßig (bounds_v1)
- Bewegung: +x_R ab Startpose, geclampt auf die 15-mm-Distanz.
- Prior-EOS: Soft-Sigmoid-Rampe am nominalen geometrischen
  Detektionszeitpunkt t_nom = r_GT,run / v — der Prior kodiert den idealen,
  latenzfreien Detektor; die Residual-GRU lernt Latenz/Overshoot/Fehlschläge.
- p_succ des Priors = 0 (Konvention wie StaticPrior im SPI-Repo; das
  Success-Residuum lernt die GRU).
- Wrench: Fz = fz_sign * F_N_mag (Vorzeichen an die realen, ins sliding_frame
  transformierten Daten anpassen!), Rest 0.

Pure torch, kein Cache, kein Multiprocessing -> Gradienten fließen zu x.
"""
import torch

from sequence_config import SequenceConfig, DT, DEFAULT_HORIZON_SAMPLES


class PilotSlidingPrior:
    def __init__(self, dt: float = DT, T: int = DEFAULT_HORIZON_SAMPLES,
                 slide_dist_m: float = 15e-3, r_gt_run_m: float = 9.25e-3,
                 fz_sign: float = -1.0, eos_softness_steps: float = 2.0):
        SequenceConfig(T, dt)
        self.dt = dt
        self.T = T
        self.slide_dist_m = slide_dist_m
        self.r_gt_run_m = r_gt_run_m
        self.fz_sign = fz_sign
        self.eos_softness = eos_softness_steps * dt

    def simulate(self, inputs_world: torch.Tensor, point_start_world: torch.Tensor,
                 max_trajectory_len: int = None, cache: bool = False) -> torch.Tensor:
        T = max_trajectory_len if max_trajectory_len is not None else self.T
        SequenceConfig(T, self.dt)
        if inputs_world.dim() == 1:
            inputs_world = inputs_world.unsqueeze(0)
            point_start_world = point_start_world.unsqueeze(0)
        B = inputs_world.size(0)
        dtype = inputs_world.dtype

        t = torch.arange(T, dtype=dtype, device=inputs_world.device).unsqueeze(0) * self.dt  # [1,T]
        v = inputs_world[:, 0:1].clamp(min=1e-6)   # [B,1] m/s
        fn = inputs_world[:, 1:2]                  # [B,1] N

        d = torch.clamp(v * t, max=self.slide_dist_m)              # [B,T] Verschiebung
        px = point_start_world[:, 0:1] + d
        py = point_start_world[:, 1:2].expand(B, T)
        pz = point_start_world[:, 2:3].expand(B, T)
        quat = point_start_world[:, 3:7].unsqueeze(1).expand(B, T, 4)

        zeros = torch.zeros(B, T, dtype=dtype, device=inputs_world.device)
        fz = self.fz_sign * fn.expand(B, T)

        # Soft-EOS am nominalen Detektionszeitpunkt t_nom = r_GT,run / v
        t_nom = self.r_gt_run_m / v                                # [B,1]
        if torch.any(t_nom.detach() > (T - 1) * self.dt):
            raise ValueError("Prior nominal event exceeds configured horizon; refusing premature forced EOS")
        eos = torch.sigmoid((t - t_nom) / self.eos_softness)       # [B,T]
        succ = zeros

        traj = torch.stack([eos, succ, px, py, pz,
                            quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3],
                            zeros, zeros, fz, zeros, zeros, zeros], dim=-1)
        # Framework-Konvention: letztes Sample hart EOS=1
        traj = traj.clone()
        traj[:, -1, 0] = 1.0
        return traj
