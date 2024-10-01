import trimesh
import pyrender
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def render_two_models_to_image_with_masks(obj_path1, obj_path2, image_path, mask1_path, mask2_path, resolution=(800, 600), max_attempts=100):
    # Load the first 3D model
    mesh1 = trimesh.load(obj_path1)

    # Randomly rotate and translate the first model
    rotate_and_translate_mesh(mesh1)

    # Create a scene with a transparent background
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])

    # Add the first mesh to the scene
    pyrender_mesh1 = pyrender.Mesh.from_trimesh(mesh1, smooth=False)
    scene.add(pyrender_mesh1)

    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = -3.0  # Adjust camera distance to fit both models
    scene.add(camera, pose=camera_pose)

    # Set up lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Create a new renderer for mask 1
    renderer1 = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    color1, depth1 = renderer1.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask1 = create_mask_from_alpha(color1)
    mask1.save(mask1_path)
    renderer1.delete()  # Clean up after rendering mask 1

    # Now load the second model
    mesh2 = trimesh.load(obj_path2)

    # Try to place the second model without overlap with the first one
    for attempt in range(max_attempts):
        # Make sure to rotate and translate the second model randomly
        rotate_and_translate_mesh(mesh2)

        # Create a temporary scene just for the second model
        scene_tmp = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])

        # Add the second mesh to the temporary scene and render it for mask generation
        pyrender_mesh2_tmp = pyrender.Mesh.from_trimesh(mesh2, smooth=False)
        scene_tmp.add(pyrender_mesh2_tmp)
        scene_tmp.add(camera, pose=camera_pose)
        scene_tmp.add(light, pose=camera_pose)

        # Create a new renderer for mask 2
        renderer2 = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
        color2_tmp, depth2_tmp = renderer2.render(scene_tmp, flags=pyrender.RenderFlags.RGBA)
        mask2_tmp = create_mask_from_alpha(color2_tmp)

        # Visualize the masks side by side
        visualize_masks(mask1, mask2_tmp)

        # Check if the masks overlap
        if not check_masks_overlap(mask1, mask2_tmp):
            print(f"Success: Found a non-overlapping placement on attempt {attempt + 1}")
            mask2_tmp.save(mask2_path)
            renderer2.delete()  # Clean up after rendering mask 2
            # Add the second model to the original scene (where both models are rendered)
            pyrender_mesh2 = pyrender.Mesh.from_trimesh(mesh2, smooth=False)
            scene.add(pyrender_mesh2)
            break
        else:
            print(f"Attempt {attempt + 1}: Overlap detected. Trying again.")
            renderer2.delete()  # Clean up before trying again

    else:
        print(f"Warning: Could not find a non-overlapping position for the second model after {max_attempts} attempts.")

    # Create a new renderer for the final image
    renderer_final = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    final_color, _ = renderer_final.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer_final.delete()  # Clean up after rendering final image

    # Convert the rendered image to a PIL image and save it
    final_image = Image.fromarray(final_color, 'RGBA')
    final_image.convert('RGB').save(image_path, 'JPEG')

def rotate_and_translate_mesh(mesh, translation_range=1.5):
    """Randomly rotate and translate a mesh."""
    rotation = np.random.uniform(0, 90, size=3)  # Random rotation for x, y, z
    rotation_matrix_x = trimesh.transformations.rotation_matrix(np.radians(rotation[0]), [1, 0, 0], point=mesh.centroid)
    rotation_matrix_y = trimesh.transformations.rotation_matrix(np.radians(rotation[1]), [0, 1, 0], point=mesh.centroid)
    rotation_matrix_z = trimesh.transformations.rotation_matrix(np.radians(rotation[2]), [0, 0, 1], point=mesh.centroid)
    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
    mesh.apply_transform(rotation_matrix)

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
    """Check if two masks overlap by comparing non-zero pixels."""
    mask1_data = np.array(mask1)
    mask2_data = np.array(mask2)

    # If there are any non-zero pixels in both masks at the same location, they overlap
    overlap = np.logical_and(mask1_data > 0, mask2_data > 0)
    return np.any(overlap)

def visualize_masks(mask1, mask2):
    """Display the two masks side by side for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(mask1, cmap='gray')
    axes[0].set_title('Mask 1')
    axes[0].axis('off')

    axes[1].imshow(mask2, cmap='gray')
    axes[1].set_title('Mask 2')
    axes[1].axis('off')

    plt.show()

# Example usage
obj_path1 = 'models/AAC/AAC_13.obj'
obj_path2 = 'models/Mortar/mortar_1.obj'

for i in range(5):
    image_path = f'images/rendered_two_models_fixed_{i+1}.jpg'
    mask1_path = f'images/mask_model1_{i+1}.jpg'
    mask2_path = f'images/mask_model2_{i+1}.jpg'
    render_two_models_to_image_with_masks(obj_path1, obj_path2, image_path, mask1_path, mask2_path)
