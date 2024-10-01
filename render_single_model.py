import trimesh
import pyrender
import numpy as np
from PIL import Image, ImageDraw

def render_single_model_to_image_with_mask(obj_path, image_path, mask_path, resolution=(800, 600)):
    # Load the 3D model
    mesh = trimesh.load(obj_path)

    # Randomly rotate the model around its centroid with 3 rotation angles (x, y, z)
    rotation = np.random.uniform(0, 90, size=3)  # Random rotation for x, y, z
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.radians(rotation[0]), [1, 0, 0], point=mesh.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(np.radians(rotation[1]), [0, 1, 0], point=mesh.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(np.radians(rotation[2]), [0, 0, 1], point=mesh.centroid)
    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
    mesh.apply_transform(rotation_matrix)

    # Randomly translate the model within a range that keeps it inside the view
    translation_range = 1.5  # Adjust this range to ensure models stay within view
    translation = np.random.uniform(-translation_range, translation_range, size=3)
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    mesh.apply_transform(translation_matrix)

    # Create a scene with a transparent background
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])

    # Add the mesh to the scene
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pyrender_mesh)

    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = -3.0  # Adjust camera distance to fit the model
    scene.add(camera, pose=camera_pose)

    # Set up lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render the scene with a transparent background
    renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    # Convert the rendered image to a PIL image
    rendered_image = Image.fromarray(color, 'RGBA')

    # Create a mask based on the alpha channel (non-transparent pixels will be white, background will be black)
    alpha_channel = color[:, :, 3]
    mask = Image.fromarray((alpha_channel > 0).astype(np.uint8) * 255, 'L')  # Create a binary mask

    # Save the mask
    mask.save(mask_path)

    # Convert to RGB (to discard the alpha channel) and save the rendered image
    rendered_image.convert('RGB').save(image_path, 'JPEG')

# Example usage
obj_path = 'models/AAC/AAC_13.obj'
image_path = 'images/rendered_single_model.jpg'
mask_path = 'images/rendered_single_model_mask.jpg'

render_single_model_to_image_with_mask(obj_path, image_path, mask_path)
