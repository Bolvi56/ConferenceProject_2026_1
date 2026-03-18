import opensim as osim
import numpy as np
import pandas as pd

class HandGraspEnv:
    def __init__(self, model_path, dt=0.001, max_time=0.5):
        self.model = osim.Model(model_path)
        self.model.buildSystem()
        
        self.controller = osim.PrescribedController()
        self.actuators = []
        self.control_funcs = []

        for comp in self.model.getComponentsList():
            if "CoordinateActuator" in comp.getClassName():
                ca = osim.CoordinateActuator.safeDownCast(comp)
                if ca is not None:
                    self.actuators.append(ca)
                    self.controller.addActuator(ca)
                    func = osim.Constant(0.0)
                    self.control_funcs.append(func)
                    self.controller.prescribeControlForActuator(ca.getName(), func)
        
        self.model.addController(self.controller)
        self.model.buildSystem()
        
        self.contact_forces_elems = {}
        force_set = self.model.getForceSet()
        for i in range(force_set.getSize()):
            f = force_set.get(i)
            if "ElasticFoundationForce" in f.getConcreteClassName():
                self.contact_forces_elems[f.getName()] = f

        self.dt = dt
        self.max_time = max_time
        self.traj = []
        self.state = self.model.initializeState()
        self.manager = None

    def reset(self):
        self.state = self.model.initializeState()
        self.time = 0.0
        self.traj = []
        
        coord_set = self.model.getCoordinateSet()
        for i in range(coord_set.getSize()):
            c = coord_set.get(i)
            c.setLocked(self.state, False)
            safe_val = c.getRangeMax() if c.getName() in ("cmc_flexion", "cmc_abduction") else 0
            c.setValue(self.state, safe_val)
            c.setSpeedValue(self.state, 0.0)
        
        self.model.realizeVelocity(self.state)
        
        self.manager = osim.Manager(self.model)
        self.manager.setIntegratorMethod(osim.Manager.IntegratorMethod_RungeKuttaMerson)
        self.manager.setIntegratorAccuracy(1e-3)
        
        self.state.setTime(0.0)
        self.manager.initialize(self.state)
        self._record_state()
        
        return self._get_state_vector()

    def _apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        for i, a in enumerate(action):
            self.control_funcs[i].setValue(float(a))

    def _record_state(self):
        coord_set = self.model.getCoordinateSet()
        row = {"time": round(self.time, 6)}
        
        for i in range(coord_set.getSize()):
            c = coord_set.get(i)
            q_val = c.getValue(self.state)
            q_val = np.clip(q_val, c.getRangeMin(), c.getRangeMax())
            row[c.getName()] = np.rad2deg(q_val)
        
        self.traj.append(row)

    def step(self, action):
        self._apply_action(action)
        t_end = self.time + self.dt
        
        try:
            self.state = self.manager.integrate(t_end)
            self.model.realizePosition(self.state)
            self.model.realizeVelocity(self.state)
            self.model.realizeDynamics(self.state)
            self.time = self.state.getTime()
        except Exception as e:
            return self._get_state_vector(), -20.0, True, {"error": str(e)}

        self._record_state()

        contact_forces = {}
        for name, force in self.contact_forces_elems.items():
            try:
                vals = force.getRecordValues(self.state)
                fx, fy, fz = float(vals.get(0)), float(vals.get(1)), float(vals.get(2))
                mag = np.linalg.norm([fx, fy, fz])
                if mag > 1e-6:
                    contact_forces[name] = {"fx": fx, "fy": fy, "fz": fz, "mag": mag}
            except:
                continue

        reward = sum(d["mag"] for d in contact_forces.values()) * 0.01 - 0.001 * np.sum(np.abs(action))
        done = self.time >= self.max_time

        return self._get_state_vector(), reward, done, {"contacts": contact_forces}

    def _get_state_vector(self):
        try:
            q = self.state.getQ().to_numpy()
            u = self.state.getU().to_numpy()
        except:
            q = np.array(self.state.getQ())
            u = np.array(self.state.getU())
        return np.concatenate([q, u]).astype(np.float32)

def export_traj_to_sto(env, file_path):
    df = pd.DataFrame(env.traj).sort_values("time").reset_index(drop=True)
    
    header = f"Coordinates\nversion=1\nnRows={len(df)}\nnColumns={len(df.columns)}\ninDegrees=yes\nendheader\n"
    
    with open(file_path, 'w') as f:
        f.write(header)
        df.to_csv(f, sep='\t', index=False, float_format="%.6f")