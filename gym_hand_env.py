import gym
import numpy as np
from gym import spaces
from hand_env import HandGraspEnv

class GymHandGraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model_path, dt=0.001, max_time=0.05):
        super().__init__()
        self.env = HandGraspEnv(model_path, dt=dt, max_time=max_time)

        # Estado: vector de floats (ya sabes su dimensión con reset)
        state = self.env.reset()
        obs_dim = state.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Acción: por ahora controla solo 5 músculos con rango suave [0, 0.05]
        self.n_ctrl_muscles = 5
        self.action_space = spaces.Box(
            low=0.0, high=0.02, shape=(self.n_ctrl_muscles,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        return state, {}

    def step(self, action):
        self._apply_action(action)

        # DEBUG: imprime tiempo y acción
        print("DEBUG time =", self.time, "action (primeros 5) =", action[:5])

        # DEBUG: mira activación y longitud de FDSI justo antes de integrar
        mset = self.model.getMuscles()
        for i in range(mset.getSize()):
            m = mset.get(i)
            if m.getName() == "FDSI":
                print("DEBUG FDSI activation=", m.getActivation(self.state))
                # si tienes estos métodos:
                # print("DEBUG FDSI fiberLength=", m.getFiberLength(self.state))
                # print("DEBUG FDSI tendonLength=", m.getTendonLength(self.state))

        t_end = self.time + self.dt
        print(f"Integrating from {self.time:.4f} to {t_end:.4f}")
        try:
            self.manager.integrate(t_end)
        except Exception as e:
            print("Integrator failed:", e)
            next_state = self._get_state_vector()
            return next_state, -10.0, True, {"error": str(e)}


    def render(self):
        pass

    def close(self):
        pass
