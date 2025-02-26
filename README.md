# CyberMicrogrid

**CyberMicrogrid** is an open-source Python library that seamlessly integrates **power grid simulation** with **cybersecurity research**, enabling researchers, engineers, and students to **simulate**, **analyze**, and **defend** cyber-physical microgrids. By offering a flexible environment for **reinforcement learning (RL)**-based control under a variety of realistic attack scenarios, CyberMicrogrid empowers users to **develop** and **evaluate** cutting-edge defense strategies for modern power systems.

---

## Key Highlights

- **Realistic Microgrid Environment**  
  The core of CyberMicrogrid is the `PowerGridEnvironment`, which accurately models a campus microgrid with **buses**, **transformers**, **lines**, **loads**, and **distributed generators**. It supports a diverse range of **cyberattack modes**—including FDI, MITM, DOS, DDOS, and many more—each configurable by attack strength.

- **Diverse Reinforcement Learning Agents**  
  Experiment with built-in RL agents spanning **value-based** (DQN, DoubleDQN, DuelingDQN) and **policy gradient** (REINFORCE, UltraFastPPO) methods. These agents serve as a foundation for customizing and extending RL-based defense strategies in complex microgrid environments.

- **Comprehensive Training & Analysis**  
  Streamlined **training routines** let you quickly run simulations, track performance metrics (e.g., **power loss**, **frequency deviation**, **feasibility rate**), and iterate over different cyberattack scenarios. An integrated **analysis module** simplifies the process of quantifying and comparing system vulnerabilities.

- **Educational & Research Tool**  
  Ideal for **university courses**, **lab experiments**, and **corporate R&D**, CyberMicrogrid provides a hands-on approach to understanding how cyber threats can disrupt critical infrastructure—and how **reinforcement learning** can mitigate these threats.

- **Modular & Extensible**  
  The library’s modular design allows for easy **integration** with other data pipelines and **custom extensions**. Whether you’re prototyping new agents or adding novel attack vectors, CyberMicrogrid’s flexible architecture supports rapid experimentation.
- **Available attacks** 

| **No.** | **Attack Mode**                 | **Brief Description**                                                                                                      |
|---------|---------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| 1       | **NONE**                        | Normal operation with no attack.                                                                                           |
| 2       | **FDI**                         | False Data Injection: Alters bus voltage measurements to mislead the system.                                               |
| 3       | **MITM**                        | Man-In-The-Middle: Intercepts and may modify data in transit without detection.                                            |
| 4       | **DOS**                         | Denial of Service: Disables selected network lines, reducing connectivity.                                                 |
| 5       | **REPLAY**                      | Replay Attack: Replays outdated (stale) data to disrupt system monitoring.                                                 |
| 6       | **DDOS**                        | Distributed DoS: Simultaneously disables multiple lines to overwhelm the system.                                           |
| 7       | **APT**                         | Advanced Persistent Threat: Gradually reduces distributed generation output.                                             |
| 8       | **RANSOMWARE**                  | Ransomware Attack: Randomizes bus voltage levels to force corrective actions.                                              |
| 9       | **PHISHING**                    | Phishing Attack: Introduces incorrect control actions by misleading operator inputs.                                       |
| 10      | **ZERO_DAY**                    | Zero-Day Exploit: Randomly toggles line statuses and perturbs bus voltages unpredictably.                                  |
| 11      | **INSIDER**                     | Insider Attack: Disables certain lines and subtly perturbs voltage to avoid detection.                                   |
| 12      | **GPS_SPOOF**                   | GPS Spoofing: Applies a uniform voltage offset across all buses, simulating location spoofing.                             |
| 13      | **SUPPLY_CHAIN**                | Supply Chain Attack: Imposes persistent, systematic voltage offsets by tampering with supply chain data.                   |
| 14      | **CONTROL_HIJACK**              | Control Hijack: Randomly toggles line statuses with increased frequency to seize network control.                        |
| 15      | **CASCADING**                   | Cascading Failure: Triggers successive line outages leading to voltage instability.                                      |
| 16      | **SENSOR_JAM**                  | Sensor Jam: Forces abnormal voltage readings by jamming sensor signals.                                                  |
| 17      | **DATA_EXFIL**                  | Data Exfiltration: No direct physical effect; impacts system reward through penalty functions.                           |
| 18      | **MALWARE**                     | Malware Attack: Randomly disrupts line status and reduces generation to degrade grid performance.                        |
| 19      | **LOAD_ALTERING**               | Load Altering: Introduces sinusoidal fluctuations to load demands, causing oscillations in grid performance.               |
| 20      | **ENERGY_STORAGE_MANIPULATION** | Energy Storage Manipulation: Reduces battery output by altering state-of-charge readings.                                  |
| 21      | **DG_TAMPERING**                | Distributed Generation Tampering: Introduces harmonic distortions in renewable generation outputs.                       |
| 22      | **SYNCHRONIZATION**             | Synchronization Attack: Alters phase angles at buses to hinder proper reconnection with the main grid.                     |
| 23      | **MARKET_MANIPULATION**         | Market Manipulation: Adjusts price signals in transactive energy systems to destabilize market operations.                 |
| 24      | **FREQUENCY_CONTROL**           | Frequency Control Attack: Alters measured frequency to trigger inappropriate generator droop responses.                    |
| 25      | **VOLTAGE_STABILITY**           | Voltage Stability Attack: Reduces reactive power injection from inverters, compromising voltage regulation.              |
| 26      | **BLACK_START**                 | Black Start Compromise: Disables black start generators by setting their output to zero during restoration periods.        |
| 27      | **COORDINATION**                | Coordination Attack: Introduces delays in communication between distributed resources, disrupting coordinated control.   |

---


## Example Use Cases

1. **Cybersecurity Research**  
   Investigate the impact of sophisticated cyberattacks on microgrid operations and devise robust RL-based defense strategies.

2. **Grid Reliability Studies**  
   Examine how disruptions—like sensor spoofing or denial-of-service attacks—affect frequency stability and power flows in real-time.

3. **Teaching & Training**  
   Use CyberMicrogrid as a hands-on educational platform to teach the fundamentals of **cyber-physical systems**, **smart grids**, and **reinforcement learning**.

---

## Getting Started

1. **Install & Import**  
   After cloning or downloading the library, install via:
   ```bash
   !git clone https://github.com/MohamedMassaoudi-code/cybermicrogrid.git
    %cd cybermicrogrid
    !pip install -e .

Then import and create a microgrid network:

## Usage Example

### 1. Import and Create a Microgrid Network

```python
from cybermicrogrid.networks import create_campus_microgrid_network

net = create_campus_microgrid_network()
```

### 2. **Configure the Environment**
```bash
from torch_geometric.data import Data
from cybermicrogrid.environment import PowerGridEnvironment
from cybermicrogrid.utils import ensure_features

data = Data(num_nodes=net.bus.shape[0])
data = ensure_features(data)

env = PowerGridEnvironment(pp_net=net, data=data, attack_mode="DOS", attack_strength=0.1)
```
### 3. **Train an Agent**

```
from cybermicrogrid.agents import DQNAgent
from cybermicrogrid.training import train_value_based_agent

agent = DQNAgent(state_dim=data.num_nodes, action_dim=len(net.line))
env, rewards, freq, power, steps, times = train_value_based_agent(
    net, data, agent, num_episodes=10, attack_mode="DOS"
)
```
**Contributing**
CyberMicrogrid is a community-driven project. We welcome contributions that enhance the simulation environment, add new RL agents, or introduce additional cyberattack scenarios. If you’d like to get involved, please:

**Fork** the repository on GitHub.
**Create** a new branch for your feature or bug fix.
**Open** a pull request with a clear description of your changes.

### **License**
This project is licensed under the MIT License—see the LICENSE file for details.

### **Contact**
For questions, feature requests, or collaboration inquiries, please open an issue on GitHub or reach out to the maintainers directly. We look forward to your feedback and contributions!

## . Citation
Use this bibtex to cite this repository:

@ARTICLE{
  author={Massaoudi. {Mohamed}, et al.},
  journal={IEEE Transactions on Industrial Cyber-Physical Systems}, 
  title={A Transformer-based Graph Actor-Critic PPO Framework for Microgrid Cyber Security Via Combined Cyber-Physical Attack Graph Analysis}, 
  year={2025},
  volume={xx},
  number={xx},
  pages={1-12}}

© Copyright 2025-2026 by Resilient Energy Systems Lab (RESLab), University of St. Thomas, Texas A&M Engineering Experiment Station (TEES), and Texas A&M University.
