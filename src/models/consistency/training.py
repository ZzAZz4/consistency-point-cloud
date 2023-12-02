
from src.models.consistency.schedule import TimeScheduler
from src.models.consistency.model import BaseParametrization, Consistency, ConsistencyOutput
from dataclasses import dataclass
import logging
import torch


@dataclass
class ConsistencyTrainingOutput:
    t_cur: torch.Tensor
    t_next: torch.Tensor
    z: torch.Tensor
    out_cur: ConsistencyOutput
    out_next: ConsistencyOutput



class ConsistencyTraining:
    def __init__(self, time_scheduler: TimeScheduler, parametrization: BaseParametrization):
        self.time_scheduler = time_scheduler
        self.parametrization = parametrization

    def __call__(self, model: Consistency, iteration: int, pos: torch.Tensor, pos_batch: torch.Tensor, par: torch.Tensor, par_batch: torch.Tensor):
        device = pos.device
        batch_size = int(pos_batch.max()) + 1
        t_cur, t_next = self.time_scheduler.get_times(iteration, batch_size, device=device)
        logging.debug(f"t_cur: {t_cur}")
        logging.debug(f"t_next: {t_next}")

        z = torch.randn_like(pos, device=device)
        logging.debug(f"z_mean: {z.mean()}")
        logging.debug(f"z_std: {z.std()}")

        x_next = pos + t_next[pos_batch, None] * z
        out_next = model(x=x_next, t=t_next, x_batch=pos_batch, par=par, par_batch=par_batch)
        logging.debug(f"out_next_mean: {out_next.output.mean()}")
        logging.debug(f"out_next_std: {out_next.output.std()}")

        with torch.no_grad():
            x_cur = pos + t_cur[pos_batch, None] * z
            out_cur = model(x=x_cur, t=t_cur, x_batch=pos_batch, par=par, par_batch=par_batch)
            logging.debug(f"out_cur_mean: {out_cur.output.mean()}")
            logging.debug(f"out_cur_std: {out_cur.output.std()}")
        
        return ConsistencyTrainingOutput(t_cur, t_next, z, out_cur, out_next)
    

        