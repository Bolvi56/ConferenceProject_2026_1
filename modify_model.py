# modify_model.py
import opensim as osim

# Ruta del modelo original
INPUT_MODEL_PATH = r"D:\RL_project\models\Geometry_flatted\Model_TRY.osim"
# Ruta del modelo de salida con actuadores
OUTPUT_MODEL_PATH = r"D:\RL_project\models\Geometry_flatted\Model_TRY_A.osim"


def add_coordinate_actuator(model, coord_name, optimal_force=5.0,
                            min_control=-1.0, max_control=1.0):
    """
    Crea y añade un CoordinateActuator para la coordinate dada.
    """
    act = osim.CoordinateActuator()
    act.setName(f"{coord_name}_actuator")
    act.set_coordinate(coord_name)
    act.setOptimalForce(optimal_force)
    act.setMinControl(min_control)
    act.setMaxControl(max_control)
    model.addForce(act)
    print(f"  + Added CoordinateActuator for coordinate '{coord_name}'")


def main():
    print("Loading model:", INPUT_MODEL_PATH)
    model = osim.Model(INPUT_MODEL_PATH)

    # Listar todas las coordinates disponibles
    coord_set = model.getCoordinateSet()
    coord_names = [coord_set.get(i).getName() for i in range(coord_set.getSize())]
    print("\nCoordinates in model:")
    for name in coord_names:
        print(" ", name)

    # === ELIGE AQUÍ LAS COORDINATES A CONTROLAR POR TORQUE ===
    # Ejemplo: suponiendo que tienes "deviation" y "flexion"
    coords_to_control = [
        "deviation",
        "flexion",
        # añade aquí otras coordinates que quieras controlar,
        # por ejemplo: "2mcp_flexion", "3mcp_flexion", etc.
    ]

    print("\nAdding CoordinateActuators to:")
    for cname in coords_to_control:
        if cname not in coord_names:
            print(f"  ! WARNING: coordinate '{cname}' not found in model, skipping")
            continue
        add_coordinate_actuator(model, cname, optimal_force=5.0)

    # Re-inicializar sistema para asegurarse de que el modelo es consistente
    print("\nRe-initializing system...")
    _ = model.initSystem()

    # Guardar modelo modificado
    print("\nSaving modified model to:", OUTPUT_MODEL_PATH)
    model.printToXML(OUTPUT_MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
