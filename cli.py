# cli.py
import argparse
from cybermicrogrid.networks import create_campus_microgrid_network
from cybermicrogrid.environment import PowerGridEnvironment
from cybermicrogrid.utils import ensure_features
from torch_geometric.data import Data

def run_attack_scenario(attack_mode, attack_strength=0.1, num_episodes=10):
    # Create the microgrid network and corresponding state
    net = create_campus_microgrid_network()
    nbuses = net.bus.shape[0]
    data = Data(num_nodes=nbuses)
    data = ensure_features(data)
    
    # Initialize the environment with the selected attack mode
    env = PowerGridEnvironment(net, data, attack_mode=attack_mode, attack_strength=attack_strength)
    
    print(f"Running simulation for attack mode: {attack_mode.upper()}")
    
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        # Simple loop: here, you can replace the action selection with a proper agent if desired.
        while steps < 20:
            # For demonstration, we use a dummy action (e.g., toggling the first line).
            action = 0  
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a cyber-physical microgrid attack scenario simulation.")
    parser.add_argument("--attack_mode", type=str, required=True,
                        help="Attack mode to simulate (e.g., NONE, FDI, MITM, DOS, etc.)")
    parser.add_argument("--attack_strength", type=float, default=0.1, help="Strength of the attack.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run.")
    args = parser.parse_args()
    run_attack_scenario(args.attack_mode, args.attack_strength, args.num_episodes)
