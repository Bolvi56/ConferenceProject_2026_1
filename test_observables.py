from hand_env import HandGraspEnv
from observable_extraction import ObservableExtractor
import numpy as np

MODEL_PATH = r"D:\RL_project\models\Geometry_flatted\Model_Contact.osim"

print("="*80)
print("PRUEBA DE OBSERVABLES Y RECOMPENSAS")
print("="*80)

env = HandGraspEnv(MODEL_PATH, dt=0.001, max_time=0.5)
state = env.reset()

extractor = ObservableExtractor(env)

print("\n📊 ESTADOS (Observaciones):\n")

print(f"S1 - Solo posiciones articulares:")
s1 = extractor.get_S1_joint_positions()
print(f"  Shape: {s1.shape}, Dtype: {s1.dtype}")
print(f"  Valores (primeros 5): {s1[:5]}\n")

print(f"S1+ - Posiciones + velocidades:")
s1_plus = extractor.get_S1_with_velocities()
print(f"  Shape: {s1_plus.shape}, Dtype: {s1_plus.dtype}\n")

print(f"S2 - Posiciones + objeto (GripForceBody):")
s2 = extractor.get_S2_with_object()
print(f"  Shape: {s2.shape}, Dtype: {s2.dtype}")
print(f"  Objeto state (últimos 6): {s2[-6:]}\n")

print(f"S3 - Posiciones + fuerzas de contacto:")
s3 = extractor.get_S3_with_contact_forces()
print(f"  Shape: {s3.shape}, Dtype: {s3.dtype}")
print(f"  Contactos (últimos 14): {s3[-14:]}\n")

print("🎮 ACCIONES:\n")
print(f"A1 - Torques articulares: {len(env.actuators)} DOF")
print(f"  Rango: [-1.0, 1.0] (normalizado)\n")

print("🏆 RECOMPENSAS:\n")

# Simula algunos pasos
total_reward_r1 = 0.0
total_reward_r2 = 0.0
step_count = 0

for step in range(10):
    action = np.random.uniform(-1, 1, len(env.actuators))
    state, _, done, info = env.step(action)
    
    r1 = extractor.compute_reward_R1(action)
    r2 = extractor.compute_reward_R2(action)
    
    total_reward_r1 += r1
    total_reward_r2 += r2
    step_count += 1
    
    if step == 0:
        print(f"R1 - Contactos + Fuerza:")
        print(f"  Step 0: R1 = {r1:.4f}")
        print(f"  Fórmula: n_contacts * 0.1 + total_force * 0.01 - action_cost\n")
        
        print(f"R2 - Grasp Quality:")
        print(f"  Step 0: R2 = {r2:.4f}")
        print(f"  Fórmula: entropy * 0.3 + force_closure * 0.4 + total_force * 0.05 - action_cost\n")
    
    if done:
        break

print(f"Promedio R1 ({step_count} steps): {total_reward_r1/step_count:.4f}")
print(f"Promedio R2 ({step_count} steps): {total_reward_r2/step_count:.4f}")

print("\n" + "="*80)
print("✅ CONFIGURACIONES LISTAS PARA EXPERIMENTOS DE RL")
print("="*80)
print("\nProyecto: 'RL for Grasp Control in Biomechanical Hand Model'")
print("\nExperimentos sugeridos:")
print("  1️⃣  ESTADO: Comparar S1 vs S1+ vs S2 vs S3")
print("  2️⃣  RECOMPENSA: Comparar R1 vs R2")
print("  3️⃣  ACCIÓN: Usar A1 (torques articulares)")
print("\nTotal de combinaciones: 4 estados × 2 recompensas = 8 experimentos")
print("="*80 + "\n")