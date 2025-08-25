import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import yaml
from genesis_env import Genesis_env
from mavlink_sim import start_mavlink_receive_thread
import genesis as gs


def main():

    # logging_level="warning"
    gs.init(logging_level="warning")
    
    with open("examples/drone/controller/config/rc_FPV_eval/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("examples/drone/controller/config/rc_FPV_eval/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)


    genesis_env = Genesis_env(
        env_config = env_config, 
        flight_config = flight_config,
    )

    device = "/dev/ttyUSB0"
    if not os.path.exists(device):
        print(f"[MAVLINK] Device {device} not found, skipping mavlink thread.")
    else :
        start_mavlink_receive_thread(device)

    while True:
        genesis_env.step()
if __name__ == "__main__" :
    main()


    