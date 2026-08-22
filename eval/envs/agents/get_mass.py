import xml.etree.ElementTree as ET
import sys

def compute_total_mass(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    total_mass = 0.0

    for mass in root.iter("mass"):
        if "value" in mass.attrib:
            total_mass += float(mass.attrib["value"])

    return total_mass


if __name__ == "__main__":
    urdf_file = sys.argv[1]
    total = compute_total_mass(urdf_file)
    print(f"Total mass from URDF: {total:.6f} kg")