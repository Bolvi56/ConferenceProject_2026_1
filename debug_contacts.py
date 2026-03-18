import opensim as osim
import numpy as np

MODEL_PATH = r"D:\RL_project\models\Geometry_flatted\Model_T.osim"
model = osim.Model(MODEL_PATH)
model.buildSystem()

state = model.initializeState()

# Configurar igual que reset()
coord_set = model.getCoordinateSet()
for i in range(coord_set.getSize()):
    c = coord_set.get(i)
    c.setLocked(state, False)
    safe_val = c.getRangeMax() if c.getName() in ("cmc_flexion", "cmc_abduction") else 0
    c.setValue(state, safe_val)
    c.setSpeedValue(state, 0.0)

model.realizeVelocity(state)

# Setup manager
manager = osim.Manager(model)
manager.setIntegratorMethod(osim.Manager.IntegratorMethod_RungeKuttaMerson)
manager.setIntegratorAccuracy(1e-1)
state.setTime(0.0)
manager.initialize(state)

print("=== DEBUG CRASH ===\n")

time = 0.0
dt = 0.001

for step in range(50):
    t_end = time + dt
    
    try:
        print(f"Step {step}: Integrando hasta t={t_end:.4f}...", end=" ")
        
        # Integrate
        state = manager.integrate(t_end)
        model.realizePosition(state)
        model.realizeVelocity(state)
        
        print(f"✓ OK", end="")
        
        # Try dynamics
        try:
            model.realizeDynamics(state)
            print(f" | Dynamics: ✓")
        except Exception as e:
            print(f" | Dynamics: ✗ ({e})")
            break
        
        time = state.getTime()
        
    except Exception as e:
        print(f"✗ CRASH")
        print(f"\n❌ Error en step {step}: {e}")
        print(f"   Time: {time:.4f}s")
        break