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
      DATA_EXFIL, MALWARE,
      ISLANDING, LOAD_ALTERING, ENERGY_STORAGE_MANIPULATION, DG_TAMPERING,
      SYNCHRONIZATION, MARKET_MANIPULATION, FREQUENCY_CONTROL, VOLTAGE_STABILITY,
      BLACK_START, COORDINATION
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

        # Existing attack modes (1-18)
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
            # New Attack Modes (19 - 28)
            elif self.attack_mode == "ISLANDING":
                # Simulate islanding by toggling the connection at the Point of Common Coupling (PCC)
                # Assume PCC is represented by bus index 0.
                self.attacked_buses = [0]
                current_status = self.pp_net.bus.at[0, "in_service"] if "in_service" in self.pp_net.bus.columns else True
                self.pp_net.bus.at[0, "in_service"] = not current_status
            elif self.attack_mode == "LOAD_ALTERING":
                # Alter load demands by adding a sinusoidal variation.
                if len(self.pp_net.load) > 0:
                    attacked = np.random.choice(self.pp_net.load.index,
                                                size=int(0.3 * len(self.pp_net.load)),
                                                replace=False)
                    for idx in attacked:
                        original = self.pp_net.load.at[idx, "p_mw"]
                        delta = self.attack_strength * math.sin(time.time())
                        self.pp_net.load.at[idx, "p_mw"] = original + delta
            elif self.attack_mode == "ENERGY_STORAGE_MANIPULATION":
                # Manipulate energy storage: reduce state-of-charge of batteries (assumed in sgen with 'battery' in name)
                for idx in self.pp_net.sgen.index:
                    if "battery" in str(self.pp_net.sgen.at[idx, "name"]).lower():
                        self.pp_net.sgen.at[idx, "p_mw"] *= (1 - self.attack_strength)
            elif self.attack_mode == "DG_TAMPERING":
                # Introduce harmonic distortion to distributed generation
                for idx in self.pp_net.sgen.index:
                    original = self.pp_net.sgen.at[idx, "p_mw"]
                    delta = self.attack_strength * math.sin(2 * time.time())
                    self.pp_net.sgen.at[idx, "p_mw"] = original * (1 + delta)
            elif self.attack_mode == "SYNCHRONIZATION":
                # Manipulate phase measurements to hinder reconnection
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.3 * len(self.pp_net.bus)),
                                            replace=False)
                for b in attacked:
                    epsilon = self.attack_strength * random.uniform(0.1, 0.5)
                    # Assume a phase angle attribute "theta" exists; if not, initialize it to 0.0
                    theta = self.pp_net.bus.at[b, "theta"] if "theta" in self.pp_net.bus.columns else 0.0
                    self.pp_net.bus.at[b, "theta"] = theta + epsilon
                    self.attacked_buses.append(b)
            elif self.attack_mode == "MARKET_MANIPULATION":
                # Manipulate price signals in transactive energy systems
                # This is a placeholder: assume a custom attribute 'price' in the network
                if hasattr(self.pp_net, "price"):
                    original_price = self.pp_net.price.get("Price_actual", 1.0)
                    manipulated_price = original_price * (1 + self.attack_strength * random.choice([-1, 1]))
                    self.pp_net.price["Price_offered"] = manipulated_price
            elif self.attack_mode == "FREQUENCY_CONTROL":
                # Manipulate measured frequency to trigger inappropriate droop responses.
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.3 * len(self.pp_net.bus)),
                                            replace=False)
                for b in attacked:
                    delta_f = self.attack_strength * random.uniform(0.1, 0.5)
                    # Here we mimic frequency measurement by altering a fictitious 'f_measured' field in res_bus.
                    self.pp_net.res_bus.at[b, "f_measured"] = 50 + delta_f
                    self.attacked_buses.append(b)
            elif self.attack_mode == "VOLTAGE_STABILITY":
                # Reduce reactive power injection from inverters during critical periods.
                attacked = np.random.choice(self.pp_net.bus.index,
                                            size=int(0.2 * len(self.pp_net.bus)),
                                            replace=False)
                for b in attacked:
                    if "q_inj" in self.pp_net.res_bus.columns:
                        original_q = self.pp_net.res_bus.at[b, "q_inj"]
                        self.pp_net.res_bus.at[b, "q_inj"] = original_q * (1 - self.attack_strength)
                    self.attacked_buses.append(b)
            elif self.attack_mode == "BLACK_START":
                # Disable black start resources by setting their generation to zero.
                for idx in self.pp_net.sgen.index:
                    if "black start" in str(self.pp_net.sgen.at[idx, "name"]).lower():
                        self.pp_net.sgen.at[idx, "p_mw"] = 0
            elif self.attack_mode == "COORDINATION":
                # Introduce variable time delays in communications between distributed resources.
                delay = self.attack_strength * random.uniform(0.5, 2.0)
                time.sleep(delay)
            else:
                # For attack modes that are not implemented or "NONE"
                pass

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
