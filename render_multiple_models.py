import trimesh
import pyrender
import numpy as np
from PIL import Image
import os
import random
from pathlib import Path
import time
import csv

# Define the class directories and their corresponding model subdirectories
class_directories = {
    "AAC": "models/AAC",
    "Asphalt": "models/Asphalt",
    "Ceramics": "models/Ceramics",
    "Concrete": "models/Concrete",
    "Mortar": "models/Mortar"
}

# Define paths for single mask directory and the CSV file
single_mask_dir = Path("data/single_masks")
csv_file_path = Path("data/mask_data.csv")
image_dir = Path("data/images")


def get_next_available_mask_id(directory):
    """Returns the next available smallest ID for the mask file in the directory in %04d format."""
    existing_files = list(directory.glob("mask_*.jpg"))
    existing_ids = [int(f.stem.split('_')[1]) for f in existing_files if f.stem.split('_')[1].isdigit()]
    next_id = min(set(range(1, len(existing_ids) + 2)) - set(existing_ids))
    return f"{next_id:04d}"


def get_next_available_image_id(directory):
    """Returns the next available smallest ID for the image file in the directory in %04d format."""
    existing_files = list(directory.glob("image_*.jpg"))
    existing_ids = [int(f.stem.split('_')[1]) for f in existing_files if f.stem.split('_')[1].isdigit()]
    next_id = min(set(range(1, len(existing_ids) + 2)) - set(existing_ids))
    return f"{next_id:04d}"


def save_mask_and_update_csv(mask_image, class_name, combined_image_fname):
    """Saves the mask image with the next available ID and updates the CSV file."""

    # Get the next available ID for the mask
    mask_id = get_next_available_mask_id(single_mask_dir)
    mask_filename = f"mask_{mask_id}.jpg"
    mask_path = single_mask_dir / mask_filename

    # Save the mask image
    mask_image.save(mask_path)

    # Update the CSV file
    csv_exists = csv_file_path.is_file()
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if the file is new
        if not csv_exists:
            writer.writerow(["combined_image_fname", "mask_fname", "class"])
        # Write the data row
        writer.writerow([combined_image_fname, mask_filename, class_name])


def load_random_model_from_class(class_dir):
    """Load a random .obj file from a class directory."""
    obj_files = [f for f in os.listdir(class_dir) if f.endswith('.obj')]
    if obj_files:
        obj_path = os.path.join(class_dir, random.choice(obj_files))
        return trimesh.load(obj_path)
    return None


def generate_image(number_of_models, resolution=(800, 600), max_attempts=100):
    print(f"Generating image with {number_of_models} models.")
    image_id = get_next_available_image_id(image_dir)
    image_fname = f"image_{image_id}.jpg"

    model_images = []  # List to store individual model images
    complete_mask = Image.new('L', resolution, 0)  # Initialize complete mask
    class_masks = {class_name: Image.new('L', resolution, 0) for class_name in class_directories.keys()}  # Initialize class masks

    # Initialize an empty scene
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])

    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = -2.0
    scene.add(camera, pose=camera_pose)

    # Set up lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    for i in range(number_of_models):
        # Randomly select a class and load model
        class_name = random.choice(list(class_directories.keys()))
        print(f"Attempting to load model {i + 1} from class {class_name}.")
        model_mesh = load_random_model_from_class(class_directories[class_name])
        if model_mesh is None:
            print(f"Skipping model from class {class_name}, no models found.")
            continue

        for attempt in range(max_attempts):
            print(f"Attempting to place model {i + 1} (Attempt {attempt + 1}).")
            rotate_and_translate_mesh(model_mesh)
            temp_scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])
            temp_scene.add(camera, pose=camera_pose)
            temp_scene.add(light, pose=camera_pose)

            pyrender_mesh_temp = pyrender.Mesh.from_trimesh(model_mesh, smooth=False)
            temp_scene.add(pyrender_mesh_temp)

            print(f"Rendering mask for model {i + 1} in temporary scene.")
            time.sleep(3)
            temp_renderer = None  # Initialize temp_renderer to None
            try:
                temp_renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
                color, _ = temp_renderer.render(temp_scene, flags=pyrender.RenderFlags.RGBA)
            except AttributeError as e:
                print(f"Error creating OffscreenRenderer: {e}. Skipping to next attempt.")
                continue  # Skip to the next attempt if there's an error
            finally:
                if temp_renderer is not None:
                    temp_renderer.delete()

            mask_temp = create_mask_from_alpha(color)
            if not check_masks_overlap(mask_temp, class_masks):
                print(f"Success: Found a non-overlapping placement for model {i + 1}.")
                model_image = Image.fromarray(color, 'RGBA')
                model_images.append(model_image)
                save_mask_and_update_csv(mask_temp, class_name, image_fname)

                # Update complete mask and class-specific mask
                complete_mask = Image.composite(mask_temp, complete_mask, mask_temp)
                class_masks[class_name] = Image.composite(mask_temp, class_masks[class_name], mask_temp)

                # Add the mesh to the main scene
                pyrender_mesh = pyrender.Mesh.from_trimesh(model_mesh, smooth=False)
                scene.add(pyrender_mesh)
                break
            else:
                print(f"Overlap detected. Trying a different placement for model {i + 1}.")
        else:
            print(f"Warning: Could not place model from {class_name} without overlap after {max_attempts} attempts.")

    # Combine all individual model images
    combine_transparent_model_images(f"{image_dir}/{image_fname}", model_images, resolution)

    # Save complete mask and class-specific masks
    complete_mask.save(f'data/full_masks/complete_mask_{image_id}.jpg')
    for class_name, mask in class_masks.items():
        mask.save(f'data/{class_name}_masks/mask_{image_id}.jpg')

    print("Rendered image and all masks saved successfully.")


def combine_transparent_model_images(path, model_images, resolution=(800, 600),):
    combined_image = Image.new('RGBA', resolution, (0, 0, 0, 255))
    for model_image in model_images:
        combined_image.alpha_composite(model_image)
    combined_image.save(path, 'PNG')
    print("Combined image saved as 'combined_image.png'.")


def rotate_and_translate_mesh(mesh, translation_range_x=2.5, translation_range_y=1.9):
    rotation = np.random.uniform(0, 90, size=3)
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.radians(rotation[0]), [1, 0, 0], point=mesh.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(np.radians(rotation[1]), [0, 1, 0], point=mesh.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(np.radians(rotation[2]), [0, 0, 1], point=mesh.centroid)
    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
    mesh.apply_transform(rotation_matrix)
    translation_x = np.random.uniform(-translation_range_x, translation_range_x, size=1)
    translation_y = np.random.uniform(-translation_range_y, translation_range_y, size=1)
    translation_matrix = np.eye(4)
    translation_matrix[0, 3] = translation_x
    translation_matrix[1, 3] = translation_y
    mesh.apply_transform(translation_matrix)


def create_mask_from_alpha(rendered_image):
    alpha_channel = rendered_image[:, :, 3]
    mask = Image.fromarray((alpha_channel > 0).astype(np.uint8) * 255, 'L')
    return mask


def check_masks_overlap(new_mask, class_masks, allowed_overlap=0.05):
    new_mask_array = np.array(new_mask)
    exceed_overlap = False
    for class_name, class_mask in class_masks.items():
        class_mask_array = np.array(class_mask)
        overlap_area = np.logical_and(class_mask_array > 0, new_mask_array > 0)
        overlap_pixels = np.sum(overlap_area)
        class_mask_area = np.sum(class_mask_array > 0)
        overlap_fraction = overlap_pixels / class_mask_area if class_mask_area > 0 else 0
        if overlap_fraction > allowed_overlap:
            exceed_overlap = True
        else:
            updated_mask = np.where(overlap_area, 0, class_mask_array).astype(np.uint8)
            class_masks[class_name] = Image.fromarray(updated_mask, 'L')
    return exceed_overlap


num_of_images = 8
for i in range(num_of_images):
    print(f"Generating image number {i+1}/{num_of_images}")
    number_of_models = random.randint(7, 11)
    generate_image(number_of_models)
