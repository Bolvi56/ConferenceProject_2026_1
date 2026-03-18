from hand_env import HandGraspEnv, export_traj_to_sto
import numpy as np
from tqdm import trange

MODEL_PATH = r"D:\RL_project\models\Geometry_flatted\Model_T.osim"

env = HandGraspEnv(MODEL_PATH, dt=0.001, max_time=0.5)
state = env.reset()

print(f"✅ Entorno inicializado\n")

for t in trange(30, desc="Simulación"):
    if t == 0:
        action = np.zeros(len(env.actuators))
    else:
        action -= 0.0001
    state, reward, done, info = env.step(action)
    
    if done:
        break

export_traj_to_sto(env, "hand_env_debug.sto")
print("✅ Simulación completada")