import trimesh
import pyrender
import numpy as np
from PIL import Image

# Path to your model
model_path = "models/AAC/AAC_13.obj"

# Load the 3D model using trimesh
mesh = trimesh.load(model_path)
print(f"Loaded model {model_path} with {len(mesh.vertices)} vertices.")

# Create a pyrender scene and add the mesh
scene = pyrender.Scene()
pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
scene.add(pyrender_mesh)

# Set up the camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.eye(4)
camera_pose[2, 3] = -2.0  # Adjust this to fit the model size
scene.add(camera, pose=camera_pose)

# Add a directional light
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=camera_pose)

# Render the scene to an image
renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)

try:
    print("Rendering scene...")
    color, depth = renderer.render(scene)
    print("Rendering completed.")
except Exception as e:
    renderer.delete()
    raise RuntimeError(f"Rendering failed: {e}")
finally:
    renderer.delete()

# Convert the rendered image to a PIL image
rendered_image = Image.fromarray(color, 'RGBA')

# Save the image
output_image_path = "rendered_model.png"
rendered_image.save(output_image_path)
print(f"Image saved to {output_image_path}")
