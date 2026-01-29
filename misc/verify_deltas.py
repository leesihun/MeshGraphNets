import h5py
import numpy as np

# Open dataset
f = h5py.File('dataset/deforming_plate.h5', 'r')

# Check sample 32 (one that had extreme deltas)
sample_id = 32
data = f[f'data/{sample_id}/nodal_data'][:]  # Shape: [8, 400, num_nodes]

print(f"Sample {sample_id} shape: {data.shape}")
print()

# Extract disp_z (feature index 5)
disp_z = data[5, :, :]  # Shape: [400, num_nodes]

# Compute deltas
deltas = disp_z[1:, :] - disp_z[:-1, :]  # Shape: [399, num_nodes]

print("=" * 60)
print("ABSOLUTE VALUES (disp_z across all timesteps)")
print("=" * 60)
print(f"Min: {disp_z.min():.6f}")
print(f"Max: {disp_z.max():.6f}")
print(f"Range: {disp_z.max() - disp_z.min():.6f}")
print()

print("=" * 60)
print("DELTAS (disp_z[t+1] - disp_z[t])")
print("=" * 60)
print(f"Min delta: {deltas.min():.6f}")
print(f"Max delta: {deltas.max():.6f}")
print(f"Mean abs delta: {np.abs(deltas).mean():.6f}")
print(f"Std delta: {deltas.std():.6f}")
print()

# Find which timesteps have the largest deltas
max_delta_per_timestep = np.abs(deltas).max(axis=1)
top_timesteps = np.argsort(max_delta_per_timestep)[-5:][::-1]

print("=" * 60)
print("TOP 5 TIMESTEPS WITH LARGEST DELTAS")
print("=" * 60)
for t in top_timesteps:
    print(f"Timestep {t} -> {t+1}: max_delta = {max_delta_per_timestep[t]:.6f}")
    # Show values before and after
    node_idx = np.argmax(np.abs(deltas[t, :]))
    print(f"  Node {node_idx}: disp_z[{t}] = {disp_z[t, node_idx]:.6f}, disp_z[{t+1}] = {disp_z[t+1, node_idx]:.6f}")
print()

# Check if total displacement builds up or oscillates
print("=" * 60)
print("CHECKING MOTION PATTERN (Node 0)")
print("=" * 60)
node_idx = 0
print(f"Node {node_idx} disp_z over selected timesteps:")
for t in [0, 1, 50, 100, 150, 200, 250, 300, 350, 399]:
    print(f"  t={t:3d}: {disp_z[t, node_idx]:8.6f}")
print()

# Check a node that has large displacement
print("=" * 60)
print("CHECKING NODE WITH MAX DISPLACEMENT")
print("=" * 60)
node_with_max = np.argmax(np.abs(disp_z[-1, :]))
print(f"Node {node_with_max} (has max final displacement):")
for t in [0, 1, 50, 100, 150, 200, 250, 300, 350, 399]:
    print(f"  t={t:3d}: {disp_z[t, node_with_max]:8.6f}")

f.close()
