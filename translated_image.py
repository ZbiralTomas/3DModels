import trimesh
import pyrender
import numpy as np
from PIL import Image

def render_model_to_image(obj_path, image_path, resolution=(800, 600), translation=(0.0, 0.0, 0.0)):
    # Load the 3D model
    mesh = trimesh.load(obj_path)

    # Ensure the mesh has texture information
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv.size == 0:
        print("No texture information found in the model.")
        return

    # Apply translation to move the model
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    mesh.apply_transform(translation_matrix)

    # Create a scene with a transparent background
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])

    # Create a mesh with the texture material and add it to the scene
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pyrender_mesh)

    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = -2.0  # Move the camera back to fit the model
    scene.add(camera, pose=camera_pose)

    # Set up lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render the scene with a transparent background
    renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    # Convert the rendered image to a PIL image
    rendered_image = Image.fromarray(color, 'RGBA')

    # Convert to RGB (to discard the alpha channel) and save as JPEG
    rendered_image.convert('RGB').save(image_path, 'JPEG')

# Example usage
obj_path = 'models/AAC/AAC_13.obj'
image_path = 'images/rendered_translated.jpg'

# Move the model 1 unit to the right (positive x-direction)
translation = (1.0, 0.0, 0.0)  # (x, y, z) translation
render_model_to_image(obj_path, image_path, translation=translation)
