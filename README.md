# cybermicrogrid

CyberMicrogrid is a comprehensive Python library designed for researchers and engineers to simulate and defend cyber-physical microgrids using state-of-the-art reinforcement learning techniques. The library provides a modular framework that integrates power grid simulation with a wide range of cyberattack scenarios, enabling users to experiment with and develop advanced grid defense strategies.

Key Features
Flexible Environment Simulation:
The core of the library is the PowerGridEnvironment, which models a campus microgrid network and supports a diverse set of cyberattack scenarios including FDI, MITM, DOS, DDOS, and many others. Users can configure the environment with different attack modes and strengths, making it ideal for both academic research and practical applications.

Agent Implementations:
CyberMicrogrid offers a variety of agent classes for both value-based and policy gradient reinforcement learning methods. Examples include DummyAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PolicyGradientAgent, and an ultra-fast PPO-based agent. These agents provide a starting point for developing custom strategies for grid defense.

Network Construction:
The library includes utilities to construct realistic microgrid networks. For instance, the create_campus_microgrid_network function builds a detailed microgrid model with buses, transformers, lines, loads, and distributed generators, serving as a realistic testbed for cyberattack simulations.

Comprehensive Training Routines:
Training functions for different reinforcement learning paradigms are provided to streamline the process of running simulations, evaluating agent performance, and iterating over different cyberattack scenarios. These routines also log key performance metrics like power loss, frequency deviation, and feasibility rates.

Utility and Analysis Tools:
A set of utility functions is included to ensure proper data preparation, such as creating valid PyTorch Geometric data objects. In addition, the library offers an analysis module to compute system performance metrics and assess the impact of various attack scenarios.

Use Cases
Cybersecurity Research:
Evaluate the resilience of microgrid systems against various cyber threats by simulating realistic attack scenarios.

Reinforcement Learning Development:
Train and test RL-based defense strategies tailored to mitigate the impacts of cyberattacks on power grids.

Educational Tool:
Serve as an educational platform for teaching the principles of cyber-physical systems security and advanced control strategies in energy systems.

Getting Started
Users can install CyberMicrogrid via pip (after publishing to PyPI) or directly from the GitHub repository. The library is well-documented and includes a comprehensive README, detailed API references, and usage examples to help users quickly integrate it into their projects.

CyberMicrogrid is open for contributions, making it an evolving project driven by community input and collaboration.

This project aims to bridge the gap between cyber-security research and power systems engineering by providing an accessible and robust simulation platform for cyber-physical microgrids.
