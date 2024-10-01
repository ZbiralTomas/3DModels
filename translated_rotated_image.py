import trimesh
import pyrender
import numpy as np
from PIL import Image, ImageDraw


def render_two_models_to_image(obj_path1, obj_path2, image_path, resolution=(800, 600), max_attempts=100):
    # Load the first 3D model
    mesh1 = trimesh.load(obj_path1)

    # Randomly rotate the first model around its centroid with 3 rotation angles (x, y, z)
    rotation1 = np.random.uniform(0, 90, size=3)  # Random rotation for x, y, z
    rotation_matrix1_x = trimesh.transformations.rotation_matrix(np.radians(rotation1[0]), [1, 0, 0],
                                                                 point=mesh1.centroid)
    rotation_matrix1_y = trimesh.transformations.rotation_matrix(np.radians(rotation1[1]), [0, 1, 0],
                                                                 point=mesh1.centroid)
    rotation_matrix1_z = trimesh.transformations.rotation_matrix(np.radians(rotation1[2]), [0, 0, 1],
                                                                 point=mesh1.centroid)
    rotation_matrix1 = np.dot(rotation_matrix1_z, np.dot(rotation_matrix1_y, rotation_matrix1_x))
    mesh1.apply_transform(rotation_matrix1)

    # Randomly translate the first model within a range that keeps it inside the view
    translation_range = 1.5  # Adjust this range to ensure models stay within view
    translation1 = np.random.uniform(-translation_range, translation_range, size=3)
    translation_matrix1 = np.eye(4)
    translation_matrix1[:3, 3] = translation1
    mesh1.apply_transform(translation_matrix1)

    # Load the second 3D model
    mesh2 = trimesh.load(obj_path2)

    # Randomly rotate the second model around its centroid with 3 rotation angles (x, y, z)
    rotation2 = np.random.uniform(0, 360, size=3)  # Random rotation for x, y, z
    rotation_matrix2_x = trimesh.transformations.rotation_matrix(np.radians(rotation2[0]), [1, 0, 0],
                                                                 point=mesh2.centroid)
    rotation_matrix2_y = trimesh.transformations.rotation_matrix(np.radians(rotation2[1]), [0, 1, 0],
                                                                 point=mesh2.centroid)
    rotation_matrix2_z = trimesh.transformations.rotation_matrix(np.radians(rotation2[2]), [0, 0, 1],
                                                                 point=mesh2.centroid)
    rotation_matrix2 = np.dot(rotation_matrix2_z, np.dot(rotation_matrix2_y, rotation_matrix2_x))
    mesh2.apply_transform(rotation_matrix2)

    # Ensure the second model does not overlap with the first model
    for _ in range(max_attempts):
        translation2 = np.random.uniform(-translation_range, translation_range, size=3)
        translation_matrix2 = np.eye(4)
        translation_matrix2[:3, 3] = translation2
        mesh2_copy = mesh2.copy()
        mesh2_copy.apply_transform(translation_matrix2)

        # Check if bounding boxes intersect with a small margin
        if not check_bounding_box_overlap(mesh1, mesh2_copy, margin=0.1):
            mesh2.apply_transform(translation_matrix2)
            break
    else:
        print("Warning: Could not find a non-overlapping position for the second model after several attempts.")

    # Create a scene with a transparent background
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])

    # Add the first mesh to the scene
    pyrender_mesh1 = pyrender.Mesh.from_trimesh(mesh1, smooth=False)
    scene.add(pyrender_mesh1)

    # Add the second mesh to the scene
    pyrender_mesh2 = pyrender.Mesh.from_trimesh(mesh2, smooth=False)
    scene.add(pyrender_mesh2)

    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = -3.0  # Adjust camera distance to fit both models
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

    # Draw bounding boxes on the rendered image
    draw = ImageDraw.Draw(rendered_image)
    draw_bounding_box_from_image(draw, rendered_image)

    # Convert to RGB (to discard the alpha channel) and save as JPEG
    rendered_image.convert('RGB').save(image_path, 'JPEG')


def check_bounding_box_overlap(mesh1, mesh2, margin=0.1):
    """Check if the bounding boxes of two meshes overlap with an optional margin."""
    min1, max1 = np.copy(mesh1.bounds[0]), np.copy(mesh1.bounds[1])
    min2, max2 = np.copy(mesh2.bounds[0]), np.copy(mesh2.bounds[1])

    # Apply margin to the bounding boxes
    min1 -= margin
    max1 += margin
    min2 -= margin
    max2 += margin

    return np.all(max1 > min2) and np.all(max2 > min1)


def draw_bounding_box_from_image(draw, image):
    """Draw the bounding box based on the rendered image."""
    img_data = np.array(image)
    non_transparent_pixels = np.any(img_data[:, :, :3] != [255, 255, 255], axis=-1)  # Detect non-white pixels

    non_transparent_indices = np.argwhere(non_transparent_pixels)
    if non_transparent_indices.size > 0:
        min_y, min_x = non_transparent_indices.min(axis=0)
        max_y, max_x = non_transparent_indices.max(axis=0)

        # Draw the bounding box
        draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=2)


# Example usage
obj_path1 = 'models/AAC/AAC_13.obj'
obj_path2 = 'models/Mortar/mortar_1.obj'

for i in range(10):
    image_path = f'images/rendered_two_models_fixed_{i + 1}.jpg'
    render_two_models_to_image(obj_path1, obj_path2, image_path)
