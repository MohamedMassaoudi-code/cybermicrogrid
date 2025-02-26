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
   pip install -e .
Then import and create a microgrid network:

```bash
from cybermicrogrid.networks import create_campus_microgrid_network
net = create_campus_microgrid_network().
Configure the Environment

```bash
from torch_geometric.data import Data
from cybermicrogrid.environment import PowerGridEnvironment
from cybermicrogrid.utils import ensure_features

data = Data(num_nodes=net.bus.shape[0])
data = ensure_features(data)

env = PowerGridEnvironment(pp_net=net, data=data, attack_mode="DOS", attack_strength=0.1)
Train an Agent

from cybermicrogrid.agents import DQNAgent
from cybermicrogrid.training import train_value_based_agent

agent = DQNAgent(state_dim=data.num_nodes, action_dim=len(net.line))
env, rewards, freq, power, steps, times = train_value_based_agent(
    net, data, agent, num_episodes=10, attack_mode="DOS"
)
Contributing
CyberMicrogrid is a community-driven project. We welcome contributions that enhance the simulation environment, add new RL agents, or introduce additional cyberattack scenarios. If you’d like to get involved, please:

Fork the repository on GitHub.
Create a new branch for your feature or bug fix.
Open a pull request with a clear description of your changes.
License
This project is licensed under the MIT License—see the LICENSE file for details.

Contact
For questions, feature requests, or collaboration inquiries, please open an issue on GitHub or reach out to the maintainers directly. We look forward to your feedback and contributions!

By bridging cybersecurity research and power systems engineering, CyberMicrogrid offers a robust platform for simulating and defending modern grids against a spectrum of cyber threats—driving innovation in smart grid resiliency and control.

vbnet
Copy

You can paste this text into your `README.md` file on GitHub to provide an attractive, well-forma
