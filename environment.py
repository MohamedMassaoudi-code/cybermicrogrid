# environment.py
import copy
import random
import math
import time
import numpy as np
import pandas as pd
import torch
import pandapower as pp

from .utils import ensure_features, _safe_tensor_edges

class PowerGridEnvironment:
    """
    Cyber-physical microgrid environment supporting multiple attack scenarios.
    
    Attack modes (case-insensitive):
      NONE, FDI, MITM, DOS, REPLAY, DDOS, APT, RANSOMWARE, PHISHING, ZERO_DAY,
      INSIDER, GPS_SPOOF, SUPPLY_CHAIN, CONTROL_HIJACK, CASCADING, SENSOR_JAM,
      DATA_EXFIL, MALWARE
    """
    def __init__(self, pp_net, data, attack_mode="none", attack_strength=0.1):
        self.attack_mode = attack_mode.upper()
        self.attack_strength = attack_strength
        self.attacked_buses = []
        self.dropped_edges = []
        self.dos_lines = []
        self.previous_bus_voltages = None
        self.supply_chain_offset = None
        self.data_exfil_penalty = 10.0

        self.pp_net_original = copy.deepcopy(pp_net)
        self.pp_net = copy.deepcopy(pp_net)

        self.original_data = ensure_features(data.clone())
        self.current_data = ensure_features(data.clone())

        self.action_space = self._build_action_space()
        self._run_power_flow()
        self._update_graph()

    def _build_action_space(self):
        return list(range(len(self.pp_net.line)))

    def _run_power_flow(self):
        try:
            pp.runpp(self.pp_net, calculate_voltage_angles=True)
            self.feasible = True
        except pp.ppException:
            self.feasible = False

    def reset(self):
        self.pp_net = copy.deepcopy(self.pp_net_original)
        self.current_data = ensure_features(self.original_data.clone())
        self.attacked_buses = []
        self.dropped_edges = []
        self.dos_lines = []
        self.previous_bus_voltages = None
        self.supply_chain_offset = None
        self._run_power_flow()
        self._update_graph()
        return self.current_data

    def step(self, action):
        line_idx = self.action_space[action]
        curr_status = self.pp_net.line.at[line_idx, "in_service"]
        self.pp_net.line.at[line_idx, "in_service"] = not curr_status

        if self.attack_mode in ["DOS", "DDOS"]:
            self.dos_lines = []
            prob = self.attack_strength if self.attack_mode == "DOS" else self.attack_strength * 1.5
            for i in self.pp_net.line.index:
                if np.random.rand() < prob:
                    self.pp_net.line.at[i, "in_service"] = False
                    self.dos_lines.append(i)

        self._update_graph()
        self._run_power_flow()

        if self.feasible:
            if self.attack_mode == "FDI":
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.3 * len(self.pp_net.bus)),
                                            replace=False)
                self.attacked_buses = attacked.tolist()
                for b in attacked:
                    self.pp_net.res_bus.at[b, "vm_pu"] += self.attack_strength
            elif self.attack_mode == "REPLAY":
                if self.previous_bus_voltages is None:
                    self.previous_bus_voltages = self.pp_net.res_bus["vm_pu"].copy()
                else:
                    stale = np.random.choice(self.pp_net.bus.index,
                                             size=int(0.3 * len(self.pp_net.bus)),
                                             replace=False)
                    self.attacked_buses = stale.tolist()
                    for b in stale:
                        self.pp_net.res_bus.at[b, "vm_pu"] = self.previous_bus_voltages.at[b]
                    self.previous_bus_voltages = self.pp_net.res_bus["vm_pu"].copy()
            elif self.attack_mode == "MITM":
                pass
            elif self.attack_mode == "DDOS":
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.3 * len(self.pp_net.bus)),
                                            replace=False)
                self.attacked_buses = attacked.tolist()
                for b in attacked:
                    self.pp_net.res_bus.at[b, "vm_pu"] += np.random.uniform(-self.attack_strength, self.attack_strength)
            elif self.attack_mode == "APT":
                for idx in self.pp_net.sgen.index:
                    self.pp_net.sgen.at[idx, "p_mw"] *= (1 - self.attack_strength)
            elif self.attack_mode == "RANSOMWARE":
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.3 * len(self.pp_net.bus)),
                                            replace=False)
                self.attacked_buses = attacked.tolist()
                for b in attacked:
                    self.pp_net.res_bus.at[b, "vm_pu"] = np.random.uniform(0.8, 1.2)
            elif self.attack_mode == "PHISHING":
                if np.random.rand() < 0.5:
                    self.pp_net.line.at[line_idx, "in_service"] = curr_status
                    wrong_action = random.choice(self.action_space)
                    wrong_status = self.pp_net.line.at[wrong_action, "in_service"]
                    self.pp_net.line.at[wrong_action, "in_service"] = not wrong_status
            elif self.attack_mode == "ZERO_DAY":
                for i in self.pp_net.line.index:
                    if np.random.rand() < self.attack_strength:
                        self.pp_net.line.at[i, "in_service"] = not self.pp_net.line.at[i, "in_service"]
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.2 * len(self.pp_net.bus)),
                                            replace=False)
                self.attacked_buses = attacked.tolist()
                for b in attacked:
                    self.pp_net.res_bus.at[b, "vm_pu"] = np.random.uniform(0.85, 1.15)
            elif self.attack_mode == "INSIDER":
                insider_lines = np.random.choice(self.pp_net.line.index,
                                                 size=int(0.2 * len(self.pp_net.line)),
                                                 replace=False)
                for i in insider_lines:
                    self.pp_net.line.at[i, "in_service"] = False
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.2 * len(self.pp_net.bus)),
                                            replace=False)
                self.attacked_buses = attacked.tolist()
                for b in attacked:
                    self.pp_net.res_bus.at[b, "vm_pu"] += self.attack_strength * np.random.uniform(-0.05, 0.05)
            elif self.attack_mode == "GPS_SPOOF":
                offset = np.random.uniform(-self.attack_strength, self.attack_strength)
                self.attacked_buses = list(self.pp_net.bus.index)
                for b in self.pp_net.bus.index:
                    self.pp_net.res_bus.at[b, "vm_pu"] += offset
            elif self.attack_mode == "SUPPLY_CHAIN":
                if self.supply_chain_offset is None:
                    self.supply_chain_offset = {b: np.random.uniform(-self.attack_strength, self.attack_strength)
                                                for b in self.pp_net.bus.index}
                self.attacked_buses = list(self.pp_net.bus.index)
                for b in self.pp_net.bus.index:
                    self.pp_net.res_bus.at[b, "vm_pu"] += self.supply_chain_offset[b]
            elif self.attack_mode == "CONTROL_HIJACK":
                for i in self.pp_net.line.index:
                    if np.random.rand() < self.attack_strength * 2:
                        self.pp_net.line.at[i, "in_service"] = not self.pp_net.line.at[i, "in_service"]
            elif self.attack_mode == "CASCADING":
                for i in self.pp_net.line.index:
                    if self.pp_net.line.at[i, "in_service"]:
                        if np.random.rand() < self.attack_strength:
                            self.pp_net.line.at[i, "in_service"] = False
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.2 * len(self.pp_net.bus)),
                                            replace=False)
                self.attacked_buses = attacked.tolist()
                for b in attacked:
                    self.pp_net.res_bus.at[b, "vm_pu"] += np.random.uniform(-self.attack_strength, self.attack_strength)
            elif self.attack_mode == "SENSOR_JAM":
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.3 * len(self.pp_net.bus)),
                                            replace=False)
                self.attacked_buses = attacked.tolist()
                for b in attacked:
                    self.pp_net.res_bus.at[b, "vm_pu"] = np.random.uniform(0.5, 0.7)
            elif self.attack_mode == "DATA_EXFIL":
                pass  # Penalty applied in reward function
            elif self.attack_mode == "MALWARE":
                malware_lines = np.random.choice(self.pp_net.line.index,
                                                 size=int(0.1 * len(self.pp_net.line)),
                                                 replace=False)
                for i in malware_lines:
                    self.pp_net.line.at[i, "in_service"] = not self.pp_net.line.at[i, "in_service"]
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.2 * len(self.pp_net.bus)),
                                            replace=False)
                self.attacked_buses = attacked.tolist()
                for b in attacked:
                    self.pp_net.res_bus.at[b, "vm_pu"] += np.random.uniform(-self.attack_strength, self.attack_strength)
                for idx in self.pp_net.sgen.index:
                    self.pp_net.sgen.at[idx, "p_mw"] *= (1 - self.attack_strength * np.random.uniform(0.5, 1.5))
        
        reward = self._calc_reward()
        done = False
        return self.current_data, reward, done

    def _update_graph(self):
        edges = []
        for idx, row in self.pp_net.line.iterrows():
            if row["in_service"]:
                fb = int(row["from_bus"])
                tb = int(row["to_bus"])
                edges.append([fb, tb])
                edges.append([tb, fb])
        if self.attack_mode == "MITM" and edges:
            undirected = list({tuple(sorted(e)) for e in edges})
            ndrop = int(len(undirected) * 0.2)
            if ndrop > 0:
                drop_idx = np.random.choice(len(undirected), size=ndrop, replace=False)
                self.dropped_edges = [undirected[i] for i in drop_idx]
                remain = [u for i, u in enumerate(undirected) if i not in drop_idx]
            else:
                self.dropped_edges = []
                remain = undirected
            new_e = []
            for ed in remain:
                new_e.append([ed[0], ed[1]])
                new_e.append([ed[1], ed[0]])
            e_idx = _safe_tensor_edges(new_e)
        else:
            e_idx = _safe_tensor_edges(edges)
        self.current_data.edge_index = e_idx

    def _calc_reward(self):
        if not getattr(self, "feasible", False):
            return -1000.0
        overload = 0.0
        if "res_line" in self.pp_net and len(self.pp_net.res_line) > 0:
            for _, ln in self.pp_net.res_line.iterrows():
                if ln["loading_percent"] > 100.0:
                    overload += (ln["loading_percent"] - 100.0)
        voltage_dev = 0.0
        if "res_bus" in self.pp_net and len(self.pp_net.res_bus) > 0:
            for _, bus_r in self.pp_net.res_bus.iterrows():
                if bus_r["vm_pu"] < 0.95 or bus_r["vm_pu"] > 1.05:
                    voltage_dev += abs(bus_r["vm_pu"] - 1.0) * 100.0
        st_bonus = self._stealth_metric()
        return st_bonus - (overload + voltage_dev)

    def _stealth_metric(self):
        if (self.current_data.edge_index is None or self.current_data.edge_index.size(1) == 0):
            return 10.0
        n_undirected = self.current_data.edge_index.shape[1] // 2
        return 10.0 - float(n_undirected)

    def compute_frequency(self):
        if not getattr(self, "feasible", False):
            return 0.0
        if "res_bus" in self.pp_net and len(self.pp_net.res_bus) > 0:
            dev = np.abs(self.pp_net.res_bus["vm_pu"] - 1.0).sum()
            freq = 50 - 2.0 * dev
            return freq
        return 0.0

    def compute_active_power(self):
        if not getattr(self, "feasible", False):
            return 0.0
        if "res_line" in self.pp_net and len(self.pp_net.res_line) > 0:
            if "p_from_mw" in self.pp_net.res_line.columns:
                return self.pp_net.res_line["p_from_mw"].abs().sum()
            elif "p_from_kw" in self.pp_net.res_line.columns:
                return self.pp_net.res_line["p_from_kw"].abs().sum() / 1000.0
        return 0.0
