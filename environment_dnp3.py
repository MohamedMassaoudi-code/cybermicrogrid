# environment_dnp3.py
import copy
from .environment import PowerGridEnvironment
from .communication import DNP3Simulator

class DNP3PowerGridEnvironment(PowerGridEnvironment):
    """
    Cyber-physical microgrid environment with integrated DNP3 communications simulation.
    
    Inherits all functionality from PowerGridEnvironment and adds support for DNP3 attack scenarios.
    
    Attack modes (case-insensitive):
      All modes from PowerGridEnvironment plus:
      DNP3_ATTACK - Simulates vulnerabilities in DNP3 communications.
    """
    def __init__(self, pp_net, data, attack_mode="none", attack_strength=0.1):
        super().__init__(pp_net, data, attack_mode, attack_strength)
        # Initialize the DNP3 simulator
        self.dnp3_simulator = DNP3Simulator()

    def step(self, action):
        # Execute the original step behavior from the parent class
        state, reward, done = super().step(action)
        
        # If the selected attack mode is DNP3_ATTACK, simulate DNP3 communication issues.
        if self.attack_mode == "DNP3_ATTACK":
            # Example: simulate a dropped DNP3 message
            self.dnp3_simulator.simulate_attack("message_drop")
            # Optionally, you can also modify the reward or state to reflect the impact of the DNP3 attack.
        
        return state, reward, done
