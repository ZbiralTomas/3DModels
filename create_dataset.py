import os
import random
from PIL import Image
import numpy as np
import trimesh
import pyrender

class ModelLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.models = self.load_model_paths()

    def load_model_paths(self):
        models = {}
        for material in os.listdir(self.base_dir):
            material_dir = os.path.join(self.base_dir, material)
            if os.path.isdir(material_dir):
                material_models = [os.path.join(material_dir, model) for model in os.listdir(material_dir) if model.endswith('.obj')]
                models[material] = {
                    'paths': material_models,
                    'probability': 1 / len(material_models) if material_models else 0
                }
        return models

    def choose_random_model(self):
        material = random.choice(list(self.models.keys()))
        model_info = self.models[material]
        model_path = random.choice(model_info['paths'])
        return model_path, material

class Background:
    def __init__(self, width=1920, height=1080, color=(0, 0, 255)):
        self.width = width
        self.height = height
        self.color = color
        self.image = Image.new('RGB', (self.width, self.height), self.color)

    def get_image(self):
        return self.image

class ModelPlacer:
    def __init__(self, background, model_loader):
        self.background = background
        self.model_loader = model_loader
        self.placed_models = []
        self.all_mask = Image.new('L', self.background.image.size, 0)
        self.material_masks = {material: Image.new('L', self.background.image.size, 0) for material in model_loader.models.keys()}

    def place_models(self, num_models=5):
        for _ in range(num_models):
            success = False
            while not success:
                model_path, material = self.model_loader.choose_random_model()
                try:
                    print(f"Attempting to render model: {model_path}")
                    rendered_image, mask = self.render_model(model_path)
                    if self.can_place_model(rendered_image):
                        self.add_model_to_background(rendered_image, mask, material)
                        success = True
                except Exception as e:
                    print(f"Error rendering model {model_path}: {e}")
                    continue

    def render_model(self, model_path):
        try:
            mesh = trimesh.load(model_path)
            print(f"Loaded model {model_path} with {len(mesh.vertices)} vertices.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")

        scene = pyrender.Scene()
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(pyrender_mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.eye(4)
        camera_pose[2, 3] = -2.0  # Adjust this to fit the model size
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(viewport_width=self.background.width, viewport_height=self.background.height)
        try:
            print("Rendering scene...")
            color, depth = renderer.render(scene)
        except Exception as e:
            renderer.delete()
            raise RuntimeError(f"Rendering failed: {e}")
        finally:
            renderer.delete()

        # Ensure the image dimensions match the background dimensions
        if color.shape[0] != self.background.height or color.shape[1] != self.background.width:
            raise ValueError(f"Rendered image dimensions {color.shape[:2]} do not match background dimensions {(self.background.height, self.background.width)}")

        rendered_image = Image.fromarray(color, 'RGBA')
        mask = Image.fromarray((depth < np.inf).astype(np.uint8) * 255, 'L')

        return rendered_image, mask

    def can_place_model(self, rendered_image):
        rendered_array = np.array(rendered_image)[:, :, 3]  # Use the alpha channel
        existing_array = np.array(self.all_mask)
        combined = rendered_array + existing_array
        return not np.any(combined > 255)

    def add_model_to_background(self, rendered_image, mask, material):
        position = self.find_random_position(rendered_image)

        # Paste the rendered model onto the background
        self.background.image.paste(rendered_image, position, rendered_image)

        # Update the all materials mask
        self.all_mask.paste(mask, position, mask)

        # Update the specific material mask
        self.material_masks[material].paste(mask, position, mask)

        # Record the placement
        self.placed_models.append((rendered_image, position))

    def find_random_position(self, rendered_image):
        # Find a random position for the model
        bg_width, bg_height = self.background.image.size
        img_width, img_height = rendered_image.size

        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(0, bg_width - img_width)
            y = random.randint(0, bg_height - img_height)
            if self.can_place_model(rendered_image):
                return (x, y)

        raise RuntimeError("Couldn't find a place to put the model without overlap.")

    def save_masks(self, output_dir):
        self.all_mask.save(os.path.join(output_dir, 'all_materials_mask.png'))
        for material, mask in self.material_masks.items():
            mask.save(os.path.join(output_dir, f'{material}_mask.png'))

def main():
    # Paths and settings
    models_dir = "models"
    output_dir = "images"

    # Initialize classes
    model_loader = ModelLoader(models_dir)
    background = Background()
    model_placer = ModelPlacer(background, model_loader)

    # Place models and save results
    model_placer.place_models(num_models=5)
    background.image.save(os.path.join(output_dir, 'final_image.png'))
    model_placer.save_masks(output_dir)

if __name__ == "__main__":
    main()
