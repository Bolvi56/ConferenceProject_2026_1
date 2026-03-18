import opensim as osim
import numpy as np
from scipy.spatial import ConvexHull

class ObservableExtractor:
    """Extrae observables para diferentes configuraciones de estado (S1, S2, S3)"""
    
    def __init__(self, env):
        self.env = env
    
    # ====== ESTADOS (S) ======
    
    def get_S1_joint_positions(self):
        """S1: Solo posiciones articulares (q)"""
        q = self.env.state.getQ().to_numpy()
        return q.astype(np.float32)
    
    def get_S1_with_velocities(self):
        """S1+: Posiciones articulares + velocidades (q, u)"""
        q = self.env.state.getQ().to_numpy()
        u = self.env.state.getU().to_numpy()
        return np.concatenate([q, u]).astype(np.float32)
    
    def get_S2_with_object(self):
        """S2: Posiciones articulares + posición/orientación del objeto"""
        q = self.env.state.getQ().to_numpy()
        
        # Intenta obtener posición del objeto (GripForceBody)
        try:
            obj_body = self.env.model.getBodySet().get("GripForceBody")
            pos = obj_body.getPositionInGround(self.env.state).to_numpy()
            orient = obj_body.getOrientationInGround(self.env.state).convertRotationToBodyFixedXYZ().to_numpy()
            obj_state = np.concatenate([pos, orient])
        except:
            obj_state = np.zeros(6)
        
        return np.concatenate([q, obj_state]).astype(np.float32)
    
    def get_S3_with_contact_forces(self):
        """S3: Posiciones articulares + magnitudes de fuerzas de contacto"""
        q = self.env.state.getQ().to_numpy()
        
        # Obtén magnitudes de fuerzas de contacto
        contact_mags = []
        for name, force in self.env.contact_forces_elems.items():
            try:
                vals = force.getRecordValues(self.env.state)
                fx = float(vals.get(0))
                fy = float(vals.get(1))
                fz = float(vals.get(2))
                mag = np.linalg.norm([fx, fy, fz])
                contact_mags.append(mag)
            except:
                contact_mags.append(0.0)
        
        contact_mags = np.array(contact_mags, dtype=np.float32)
        return np.concatenate([q, contact_mags]).astype(np.float32)
    
    # ====== RECOMPENSAS (R) ======
    
    def compute_reward_R1(self, action):
        """
        R1: Recompensa basada en contactos y fuerza normal
        - Premia: número de contactos activos + magnitud total de fuerza
        - Penaliza: costo de acción
        """
        contact_forces = {}
        for name, force in self.env.contact_forces_elems.items():
            try:
                vals = force.getRecordValues(self.env.state)
                fx = float(vals.get(0))
                fy = float(vals.get(1))
                fz = float(vals.get(2))
                mag = np.linalg.norm([fx, fy, fz])
                if mag > 1e-6:
                    contact_forces[name] = mag
            except:
                continue
        
        n_contacts = len(contact_forces)
        total_force = sum(contact_forces.values())
        action_cost = 0.001 * np.sum(np.abs(action))
        
        reward = n_contacts * 0.1 + total_force * 0.01 - action_cost
        return float(reward)
    
    def compute_reward_R2(self, action):
        """
        R2: Recompensa basada en métricas de calidad de agarre
        - Premia: distribución uniforme de fuerzas + fuerza total
        - Penaliza: costo de acción
        """
        contact_forces = {}
        force_vectors = []
        
        for name, force in self.env.contact_forces_elems.items():
            try:
                vals = force.getRecordValues(self.env.state)
                fx = float(vals.get(0))
                fy = float(vals.get(1))
                fz = float(vals.get(2))
                mag = np.linalg.norm([fx, fy, fz])
                
                if mag > 1e-6:
                    contact_forces[name] = mag
                    force_vectors.append([fx, fy, fz])
            except:
                continue
        
        # Métrica 1: Distribución de fuerzas (entropía normalizada)
        if len(contact_forces) > 0:
            mags = np.array(list(contact_forces.values()))
            norm_mags = mags / mags.sum()
            entropy = -np.sum(norm_mags[norm_mags > 0] * np.log(norm_mags[norm_mags > 0] + 1e-8))
            entropy = entropy / np.log(len(mags)) if len(mags) > 1 else 0.0
        else:
            entropy = 0.0
        
        # Métrica 2: Convex hull (fuerza closure simplificado)
        force_closure = 0.0
        if len(force_vectors) >= 4:
            try:
                hull = ConvexHull(np.array(force_vectors))
                # Si el volumen es > 0, hay fuerza closure
                force_closure = 1.0 if hull.volume > 0.001 else 0.5
            except:
                force_closure = 0.0
        
        # Métrica 3: Fuerza total
        total_force = sum(contact_forces.values())
        
        action_cost = 0.001 * np.sum(np.abs(action))
        
        reward = (entropy * 0.3 + force_closure * 0.4 + total_force * 0.05) - action_cost
        return float(reward)
    
    def get_contact_info(self):
        """Retorna diccionario con info de contactos"""
        contact_forces = {}
        for name, force in self.env.contact_forces_elems.items():
            try:
                vals = force.getRecordValues(self.env.state)
                fx = float(vals.get(0))
                fy = float(vals.get(1))
                fz = float(vals.get(2))
                mag = np.linalg.norm([fx, fy, fz])
                
                if mag > 1e-6:
                    contact_forces[name] = {"fx": fx, "fy": fy, "fz": fz, "mag": mag}
            except:
                continue
        
        return contact_forces