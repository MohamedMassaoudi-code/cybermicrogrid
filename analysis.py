# analysis.py
import numpy as np
from .networks import create_campus_microgrid_network

def compute_power_loss(net):
    """
    Computes total power loss (in kW) and its percentage relative to total load.
    Assumes pp.runpp has been executed and that net.res_line contains "pl_mw"
    and net.load contains "p_mw".
    """
    if "pl_mw" in net.res_line.columns and len(net.res_line) > 0 and "p_mw" in net.load.columns and len(net.load) > 0:
        total_loss_kw = net.res_line["pl_mw"].sum() * 1000  # Convert MW to kW
        total_load_kw = net.load["p_mw"].sum() * 1000
        loss_percent = (total_loss_kw / total_load_kw) * 100 if total_load_kw != 0 else 0
        return total_loss_kw, loss_percent
    else:
        return None, None

class AttackAnalysis:
    def __init__(self, env, rewards_history, freq_history, power_history, feasible_history):
        self.env = env
        self.rewards = np.array(rewards_history)
        self.freq   = np.array(freq_history)
        self.power  = np.array(power_history)
        self.feasible_history = np.array(feasible_history)

    def analyze_system_violations(self):
        self.env._run_power_flow()
        if not getattr(self.env, "feasible", False):
            return {"voltage_violations": 0, "line_violations": 0, "total_violations": 0}
        vvio = sum(1 for _, row in self.env.pp_net.res_bus.iterrows()
                   if row["vm_pu"] < 0.95 or row["vm_pu"] > 1.05)
        lvio = sum(1 for _, row in self.env.pp_net.res_line.iterrows()
                   if row["loading_percent"] > 100.0)
        return {"voltage_violations": vvio, "line_violations": lvio, "total_violations": vvio + lvio}

    def summary_metrics(self):
        loss_kw, loss_percent = compute_power_loss(self.env.pp_net)
        mean_f  = self.freq.mean() if len(self.freq) > 0 else None
        mean_p  = self.power.mean() if len(self.power) > 0 else None
        viol    = self.analyze_system_violations()
        feasible_rate = (self.feasible_history.sum() / len(self.feasible_history)) * 100 if len(self.feasible_history) > 0 else 0
        return {
            "Power Loss (kW)": loss_kw,
            "Power Loss (%)": loss_percent,
            "Mean Frequency (Hz)": mean_f,
            "Mean Active Power (MW)": mean_p,
            "Total Violations": viol["total_violations"],
            "Feasibility Rate (%)": feasible_rate
        }
