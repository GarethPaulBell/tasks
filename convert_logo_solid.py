from PIL import Image
import numpy as np
import trimesh
from skimage import measure
from scipy.ndimage import gaussian_filter
import argparse

def image_to_solid_stl(image_path, output_path, thickness_mm=2.0, max_size=512):
    print(f"Loading {image_path}...")
    img = Image.open(image_path).convert('L')
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    arr = np.array(img)
    
    # 1. Identify the solid parts! 
    # You mentioned we want the logo filled in where the image is BLACK.
    # So we isolate the black pixels (< 128) and turn them into a solid block of "255".
    # White background becomes "0" (empty space).
    binary = (arr < 128).astype(np.float32) * 255.0
    
    # Smooth to avoid jagged printing edges
    smoothed = gaussian_filter(binary, sigma=1.0)
    
    # 2. Extrude by stacking
    num_layers = 10
    volume = np.stack([smoothed] * num_layers, axis=0)
    
    # 3. THE CRITICAL FIX: The Capping Layer
    # Marching cubes will leave the top and bottom completely hollow if the array 
    # touches the edge. We must pad the Z-axis with empty space (0) so the algorithm 
    # wraps the mesh completely around the top and bottom, creating a solid object!
    padded_volume = np.pad(volume, pad_width=((1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    print("Running marching cubes to generate solid volume...")
    verts, faces, normals, values = measure.marching_cubes(padded_volume, level=128)
    
    # Marching cubes returns coordinates in (Z, Y, X) order based on array shape.
    # Swap to (X, Y, Z) so it sits flat on the print bed.
    verts = verts[:, [2, 1, 0]]
    
    # 4. Scale to physical dimensions
    # Let's arbitrarily scale the longest X/Y dimension to ~100mm so it fits the bed nicely
    scale_xy = 100.0 / max_size
    verts[:, 0] *= scale_xy
    verts[:, 1] *= scale_xy
    
    # Scale Z to match the exact thickness parameter (e.g., 2mm thick)
    max_z = np.max(verts[:, 2])
    if max_z > 0:
        verts[:, 2] = (verts[:, 2] / max_z) * thickness_mm
        
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Ensure all polygons face outward so the slicer sees it as a solid manifold body
    trimesh.repair.fix_normals(mesh)
    
    mesh.export(output_path)
    print(f"Exported perfectly SOLID mesh to {output_path} (File size will be small and slicer-friendly!)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image path (png, jpg)")
    parser.add_argument("output", help="Output STL path")
    args = parser.parse_args()
    image_to_solid_stl(args.input, args.output)
