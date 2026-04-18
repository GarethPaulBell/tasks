from PIL import Image, ImageOps
import numpy as np
import trimesh
from skimage import measure
import argparse

def image_to_stl(image_path, output_path, thickness=2.0, threshold=128, max_size=512):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize the image to lower the resolution (this drastically reduces poly count)
    # We maintain the aspect ratio using thumbnail
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    print(f"Downsampled image to: {img.size}")
    
    # Invert if necessary (we want the logo to be white on black for extrusion)
    # img = ImageOps.invert(img)
    
    # Convert to numpy array
    arr = np.array(img)
    
    # Apply a slight Gaussian blur to smooth the edges (reduces noisy geometry)
    from scipy.ndimage import gaussian_filter
    arr = gaussian_filter(arr, sigma=1.0)
    
    # Create a 3D volume by repeating the 2D slice
    layers = int(thickness * 5) # Reduced layer density
    volume = np.stack([arr] * layers, axis=0)
    
    # Run marching cubes to extract the surface mesh
    verts, faces, normals, values = measure.marching_cubes(volume, level=threshold)
    
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Optional: Apply trimesh decimation if it's still too large
    # But usually downsampling the 2D image is enough and cleaner
    
    # Export to STL
    mesh.export(output_path)
    print(f"Exported optimized mesh to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image path (png, jpg)")
    parser.add_argument("output", help="Output STL path")
    args = parser.parse_args()
    image_to_stl(args.input, args.output)
