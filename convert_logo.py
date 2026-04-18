from PIL import Image, ImageOps
import numpy as np
import trimesh
from skimage import measure
import argparse

def image_to_stl(image_path, output_path, thickness=2.0, threshold=128):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Invert if necessary (we want the logo to be white on black for extrusion)
    # img = ImageOps.invert(img)
    
    # Convert to numpy array
    arr = np.array(img)
    
    # Create a 3D volume by repeating the 2D slice
    # This gives us the 'thickness'
    layers = int(thickness * 10) # arbitrary scaling for voxel depth
    volume = np.stack([arr] * layers, axis=0)
    
    # Run marching cubes to extract the surface mesh
    # We look for the boundary defined by the threshold
    verts, faces, normals, values = measure.marching_cubes(volume, level=threshold)
    
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Export to STL
    mesh.export(output_path)
    print(f"Exported to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image path (png, jpg)")
    parser.add_argument("output", help="Output STL path")
    args = parser.parse_args()
    image_to_stl(args.input, args.output)
