import trimesh
import pyrender
import numpy as np
from PIL import Image
import os
import random
import time

# TODO: zrušit posunutí v z, ověřit že se umísťujou obrázky všude a nevyčuhujou, konfigurace kamery shora zepředu,
# TODO: povolit překryv (maximálně 5%), náhodný výběr pozadí, modely 





# Define the class directories and their corresponding model subdirectories
class_directories = {
    "AAC": "models/AAC",
    "Asphalt": "models/Asphalt",
    "Ceramics": "models/Ceramics",
    "Concrete": "models/Concrete",
    "Mortar": "models/Mortar"
}


def load_random_model_from_class(class_dir):
    """Load a random .obj file from a class directory."""
    print(f"Loading random model from class directory: {class_dir}")
    obj_files = [f for f in os.listdir(class_dir) if f.endswith('.obj')]
    if obj_files:
        obj_path = os.path.join(class_dir, random.choice(obj_files))
        print(f"Selected model: {obj_path}")
        return trimesh.load(obj_path)
    print(f"No models found in {class_dir}")
    return None


def generate_image(number_of_models, resolution=(800, 600), max_attempts=100):
    print(f"Generating image with {number_of_models} models.")

    # Create black images for masks
    complete_mask = Image.new('L', resolution, 0)
    class_masks = {class_name: Image.new('L', resolution, 0) for class_name in class_directories.keys()}

    # Initialize an empty scene
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])

    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = -2.0  # Adjust camera distance to fit all models
    scene.add(camera, pose=camera_pose)

    # Set up lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    for i in range(number_of_models):
        # Randomly select a class
        class_name = random.choice(list(class_directories.keys()))
        print(f"Attempting to load model {i + 1} from class {class_name}.")
        model_mesh = load_random_model_from_class(class_directories[class_name])

        if model_mesh is None:
            print(f"Skipping model from class {class_name}, no models found.")
            continue  # Skip if no model found

        for attempt in range(max_attempts):
            print(f"Attempting to place model {i + 1} (Attempt {attempt + 1}).")

            # Rotate and translate the mesh
            rotate_and_translate_mesh(model_mesh)

            # Create a temporary scene for mask checking
            temp_scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])
            temp_scene.add(camera, pose=camera_pose)
            temp_scene.add(light, pose=camera_pose)

            pyrender_mesh_temp = pyrender.Mesh.from_trimesh(model_mesh, smooth=False)
            temp_scene.add(pyrender_mesh_temp)

            # Render in the temporary scene to get mask
            print(f"Rendering mask for model {i + 1} in temporary scene.")
            time.sleep(3)
            try:
                temp_renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
                color, _ = temp_renderer.render(temp_scene, flags=pyrender.RenderFlags.RGBA)
                mask_temp = create_mask_from_alpha(color)
            finally:
                temp_renderer.delete()

            # Check for overlap with the general mask
            if not check_masks_overlap(complete_mask, mask_temp):
                print(f"Success: Found a non-overlapping placement for model {i + 1}.")
                # Add the mesh to the main scene
                pyrender_mesh = pyrender.Mesh.from_trimesh(model_mesh, smooth=False)
                scene.add(pyrender_mesh)

                # Update complete mask and class-specific mask
                complete_mask = Image.composite(mask_temp, complete_mask, mask_temp)
                class_masks[class_name] = Image.composite(mask_temp, class_masks[class_name], mask_temp)
                break  # Found a non-overlapping position
            else:
                print(f"Overlap detected. Trying a different placement for model {i + 1}.")
        else:
            print(f"Warning: Could not place model from {class_name} without overlap after {max_attempts} attempts.")

    # Create a new renderer for final image generation
    print("Rendering final image.")
    time.sleep(3)
    try:
        final_renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
        final_color, _ = final_renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        final_renderer.delete()

        # Convert the rendered image to a PIL image
    final_image = Image.fromarray(final_color, 'RGBA')
    final_image.convert('RGB').save('rendered_image.jpg', 'JPEG')

    # Save masks
    print("Saving masks.")
    complete_mask.save('complete_mask.png')
    for class_name, mask in class_masks.items():
        mask.save(f'mask_{class_name}.png')

    print("Rendered image and masks saved successfully.")


def rotate_and_translate_mesh(mesh, translation_range=1.5):
    """Randomly rotate and translate a mesh."""
    rotation = np.random.uniform(0, 90, size=3)  # Random rotation for x, y, z
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.radians(rotation[0]), [1, 0, 0], point=mesh.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(np.radians(rotation[1]), [0, 1, 0], point=mesh.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(np.radians(rotation[2]), [0, 0, 1], point=mesh.centroid)
    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
    mesh.apply_transform(rotation_matrix)


    # resolve z translation
    # Randomly translate the model
    translation = np.random.uniform(-translation_range, translation_range, size=3)
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    mesh.apply_transform(translation_matrix)


def create_mask_from_alpha(rendered_image):
    """Create a mask based on the alpha channel (non-transparent pixels will be white, background will be black)."""
    alpha_channel = rendered_image[:, :, 3]
    mask = Image.fromarray((alpha_channel > 0).astype(np.uint8) * 255, 'L')  # Create a binary mask
    return mask


def check_masks_overlap(mask1, mask2):
    """Check if two masks overlap."""
    mask1_array = np.array(mask1)
    mask2_array = np.array(mask2)
    overlap = np.logical_and(mask1_array > 0, mask2_array > 0)
    return np.any(overlap)


# Example usage: Generate an image with 3 models
generate_image(7)
