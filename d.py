import os
import shutil

# ============================================================
# FIX DATASET STRUCTURE
# sample_0001/simulation.npz
# ->
# simulation_0001.npz
# ============================================================

base_dir = "dataset"

for bc_type in ["dirichlet", "neumann"]:

    bc_path = os.path.join(base_dir, bc_type)

    if not os.path.exists(bc_path):
        continue

    sample_folders = sorted(os.listdir(bc_path))

    for folder in sample_folders:

        folder_path = os.path.join(bc_path, folder)

        if not os.path.isdir(folder_path):
            continue

        old_file = os.path.join(folder_path, "simulation.npz")

        if not os.path.exists(old_file):
            continue

        # Extract number from sample_0001
        number = folder.split("_")[-1]

        new_file = os.path.join(
            bc_path,
            f"simulation_{number}.npz"
        )

        # Move file
        shutil.move(old_file, new_file)

        # Remove empty folder
        os.rmdir(folder_path)

        print(f"Moved -> {new_file}")

print("\nDONE.")