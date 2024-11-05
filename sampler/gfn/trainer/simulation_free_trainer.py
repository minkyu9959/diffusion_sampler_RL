import torch

from ..buffer import StateTransitionReplayBuffer
from .base_trainer import BaseTrainer
from ..loss import annealed_db_on_states


class SimulationFreeTrainer(BaseTrainer):
    def initialize(self):
        self.buffer = StateTransitionReplayBuffer(
            buffer_size=self.train_cfg.buffer_size,
            device=self.model.device,
            batch_size=self.train_cfg.batch_size,
            data_ndim=self.energy_function.ndim,
            num_time_steps=self.model.trajectory_length + 1,
        )
        self.backward_simulation = self.train_cfg.get("backward_simulation", False)

        if self.backward_simulation:
            self.sample_per_source = self.train_cfg.simulation_sample_size // 2
        else:
            self.sample_per_source = self.train_cfg.simulation_sample_size

    def train_step(self) -> dict:
        self.model.zero_grad()

        if self.must_sample_trajectory(self.current_epoch):
            trajectory = self.generate_trajectory()
            self.save_trajectory_to_buffer(trajectory)

        # sample the state transitions in the buffer
        states, idx = self.buffer.sample()

        cur_state, next_state = states[:, 0, :], states[:, 1, :]

        time = idx / self.model.trajectory_length
        next_time = (idx + 1) / self.model.trajectory_length

        params = self.model.forward_conditional.params(cur_state, time.unsqueeze(1))
        log_pfs = self.model.forward_conditional.log_prob(next_state, params)

        params = self.model.backward_conditional.params(
            next_state, next_time.unsqueeze(1)
        )
        log_pbs = self.model.backward_conditional.log_prob(cur_state, params)

        logZ_ratio = self.model.logZ_ratio[idx].squeeze()

        cur_state_reward = -self.model.annealed_energy.energy(time, cur_state)
        next_state_reward = -self.model.annealed_energy.energy(next_time, next_state)

        loss = annealed_db_on_states(
            log_pfs, log_pbs, logZ_ratio, cur_state_reward, next_state_reward
        )

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
        }

    def must_sample_trajectory(self, epoch: int) -> bool:
        return epoch % self.train_cfg.sample_interval == 0

    @torch.no_grad()
    def generate_trajectory(self):
        init_states = self.model.generate_initial_state(self.sample_per_source)
        traj, _, _ = self.model.get_forward_trajectory(init_states)

        if self.backward_simulation:
            bwd_traj, _, _ = self.model.get_backward_trajectory(traj[:, -1])
            traj = torch.cat([traj, bwd_traj], dim=0)

        return traj

    def save_trajectory_to_buffer(self, traj: torch.Tensor):
        states = torch.stack(
            [
                traj[:, :-1].reshape(-1, self.model.sample_dim),
                traj[:, 1:].reshape(-1, self.model.sample_dim),
            ],
            dim=1,
        )

        time = torch.arange(
            self.model.trajectory_length, device=self.model.device
        ).repeat(self.train_cfg.simulation_sample_size)

        self.buffer.add(states, time)
