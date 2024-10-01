import trimesh
import pyrender
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def render_model_to_image(obj_path, image_path, resolution=(800, 600), rotation=(0, 0, 0)):
    # Load the 3D model
    mesh = trimesh.load(obj_path)

    # Ensure the mesh has texture information
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv.size == 0:
        print("No texture information found in the model.")
        return

    print(f"Loaded mesh with {len(mesh.vertices)} vertices.")

    # Apply rotations
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.radians(rotation[0]), [1, 0, 0])
    rotation_matrix_y = trimesh.transformations.rotation_matrix(np.radians(rotation[1]), [0, 1, 0])
    rotation_matrix_z = trimesh.transformations.rotation_matrix(np.radians(rotation[2]), [0, 0, 1])

    mesh.apply_transform(rotation_matrix_x)
    mesh.apply_transform(rotation_matrix_y)
    mesh.apply_transform(rotation_matrix_z)

    print(f"Applied rotation: {rotation}")

    # Create a scene
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])

    # Create a mesh with the texture material
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pyrender_mesh)

    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = -2.0  # Move the camera back to fit the model
    scene.add(camera, pose=camera_pose)

    # Ensure the camera can see the whole model
    bounding_box = mesh.bounds
    size = np.max(bounding_box[1] - bounding_box[0])
    camera_pose[:3, 3] = [0, 0, -size * 2]

    print("Camera and light added to the scene.")

    # Set up lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render the scene with a transparent background
    renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    # Convert the rendered image to a PIL image
    rendered_image = Image.fromarray(color, 'RGBA')

    # Display the image with a transparent background
    plt.imshow(rendered_image)
    plt.axis('off')
    plt.show()

    # Convert to RGB (to discard the alpha channel) and save as JPEG
    rendered_image.convert('RGB').save(image_path, 'JPEG')
    print(f"Image saved to {image_path}")


# Example usage
obj_path = 'models/model.obj'
image_path = 'ceramics_2_rendered.jpg'
rotation = (45, 30, 60)  # Rotate 45 degrees around x-axis, 30 degrees around y-axis, 60 degrees around z-axis
render_model_to_image(obj_path, image_path, rotation=rotation)
