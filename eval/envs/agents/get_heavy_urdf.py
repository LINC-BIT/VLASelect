import xml.etree.ElementTree as ET
import sys
import os

def scale_urdf_inertia(input_path, output_path, scale_factor):
    """
    将URDF中的 mass 和 inertia 按比例放大
    """

    tree = ET.parse(input_path)
    root = tree.getroot()

    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue

        # 处理 mass
        mass = inertial.find("mass")
        if mass is not None and "value" in mass.attrib:
            old_mass = float(mass.attrib["value"])
            new_mass = old_mass * scale_factor
            mass.attrib["value"] = str(new_mass)

        # 处理 inertia
        inertia = inertial.find("inertia")
        if inertia is not None:
            for key in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
                if key in inertia.attrib:
                    old_val = float(inertia.attrib[key])
                    new_val = old_val * scale_factor
                    inertia.attrib[key] = str(new_val)

    tree.write(output_path)
    print(f"Scaled URDF saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scale_urdf.py input.urdf output.urdf scale_factor")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    scale = float(sys.argv[3])

    scale_urdf_inertia(input_file, output_file, scale)