import opensim as osim

INPUT_MODEL = r"D:\RL_project\models\Geometry_flatted\Model_TRY.osim"
OUTPUT_MODEL = r"D:\RL_project\models\Geometry_flatted\Model_T.osim"

model = osim.Model(INPUT_MODEL)
state = model.initSystem()

coord_set = model.getCoordinateSet()

# 🔥 Coordenadas seleccionadas (pulgar + dedos 2 y 3)
VALID_COORDS = [
    "deviation", "flexion", "wrist_hand_r1", "wrist_hand_r2",
    # Pulgar
    "cmc_flexion", "cmc_abduction", "mp_flexion", "ip_flexion",

    # Dedo 2
    "2mcp_abduction", "2mcp_flexion", "2pm_flexion", "2md_flexion",

    # Dedo 3
    "3mcp_abduction", "3mcp_flexion", "3pm_flexion", "3md_flexion",

    # Dedo 4 
    "4mcp_abduction", "4mcp_flexion", "4pm_flexion", "4md_flexion",

    # Dedo 5
    "5mcp_abduction", "5mcp_flexion", "5pm_flexion", "5md_flexion",
]

# 🔥 Torque base (N·m)
def get_optimal_force(name):
    if "mcp" in name:
        return 4.4
    elif "pm" in name:   # PIP
        return 2.3
    elif "md" in name:   # DIP
        return 0.72
    elif "thumb" in name or name in ["cmc_flexion", "cmc_abduction", "mp_flexion", "ip_flexion"]:
        return 2.0
    else:
        return 1.0  # abduction

added = []

print("\n=== AGREGANDO ACTUADORES FILTRADOS ===\n")

for i in range(coord_set.getSize()):
    coord = coord_set.get(i)
    name = coord.getName()

    if name not in VALID_COORDS:
        print(f"[SKIP] {name}")
        continue

    actuator = osim.CoordinateActuator()
    actuator.setName(name + "_act")
    actuator.setCoordinate(coord)

    opt_force = get_optimal_force(name)
    actuator.setOptimalForce(opt_force)

    actuator.setMinControl(-1.0)
    actuator.setMaxControl(1.0)

    model.addForce(actuator)
    added.append(name)

    print(f"[ADD] {name} | torque_max ≈ {opt_force} Nm")

# 🔥 MUY IMPORTANTE
model.finalizeConnections()

# Guardar modelo
model.printToXML(OUTPUT_MODEL)

print("\n=== RESUMEN ===")
print(f"Actuadores agregados: {len(added)}")
print("Modelo guardado en:", OUTPUT_MODEL)