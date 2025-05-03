import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
import os

def add_dust(image, amount):
    dust = np.zeros_like(image)
    h, w = dust.shape[:2]
    for _ in range(amount):
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        radius = random.randint(1, 3)
        color = (255, 255, 255) if random.random() > 0.5 else (0, 0, 0)
        cv2.circle(dust, (x, y), radius, color, -1)
    return cv2.addWeighted(image, 1, dust, 0.5, 0)

def add_scratches(image, amount):
    scratches = np.zeros_like(image)
    h, w = scratches.shape[:2]
    for _ in range(amount):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = x1 + random.randint(-30, 30), y1 + random.randint(20, 100)
        color = (255, 255, 255) if random.random() > 0.5 else (0, 0, 0)
        thickness = random.randint(1, 2)
        cv2.line(scratches, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(image, 1, scratches, 0.3, 0)

def fade_colors(image, lightness_reduction, blue_shift):
    faded = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(faded)
    l = cv2.add(l, -lightness_reduction)
    b = cv2.add(b, blue_shift)
    faded = cv2.merge((l, a, b))
    return cv2.cvtColor(faded, cv2.COLOR_LAB2BGR)

def add_noise(image, amount):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = cv2.addWeighted(image, 1 - amount, noise, amount, 0)
    return noisy

def fray_edges(image, num_cuts):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    w, h = pil_img.size
    mask = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(mask)

    for _ in range(num_cuts):
        x = random.randint(0, w)
        y = random.choice([random.randint(0, 20), random.randint(h-20, h)])
        r = random.randint(5, 20)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=0)

    for _ in range(num_cuts):
        y = random.randint(0, h)
        x = random.choice([random.randint(0, 20), random.randint(w-20, w)])
        r = random.randint(5, 20)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=0)

    pil_img.putalpha(mask)
    background = Image.new("RGB", (w, h), (235, 224, 200))
    background.paste(pil_img, mask=pil_img.split()[3])
    return cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)

def add_water_stains(image, amount):
    stains = np.zeros_like(image)
    h, w = stains.shape[:2]
    for _ in range(amount):
        x = random.randint(0, w)
        y = random.randint(0, h)
        radius = random.randint(40, 120)
        color = (220, 220, 220)
        cv2.circle(stains, (x, y), radius, color, -1)
    stains = cv2.GaussianBlur(stains, (51, 51), 0)
    return cv2.addWeighted(image, 1, stains, 0.2, 0)

def add_fold_lines(image, amount):
    folds = np.zeros_like(image)
    h, w = folds.shape[:2]
    for _ in range(amount):
        if random.random() > 0.5:
            x = random.randint(w // 4, 3 * w // 4)
            cv2.line(folds, (x, 0), (x, h), (200, 200, 200), 2)
        else:
            y = random.randint(h // 4, 3 * h // 4)
            cv2.line(folds, (0, y), (w, y), (200, 200, 200), 2)
    folds = cv2.GaussianBlur(folds, (11, 11), 0)
    return cv2.addWeighted(image, 1, folds, 0.3, 0)


def add_color_cast(image, intensity_range=(10, 50)):
    """Add a random color cast to the image."""
    h, w = image.shape[:2]
    
    # Choose a random cast color (RGB)
    cast_color = np.array([
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ], dtype=np.uint8)

    # Create an image full of the cast color
    color_layer = np.full((h, w, 3), cast_color, dtype=np.uint8)

    # Blend it with the original image
    intensity = random.uniform(intensity_range[0]/100, intensity_range[1]/100)
    blended = cv2.addWeighted(image, 1 - intensity, color_layer, intensity, 0)

    return blended

def simulate_scan_artifacts(image, damage_level="normal"):
    """Apply artifacts with random intensity based on damage level."""

    if damage_level == "normal":
        lightness_reduction = random.randint(10, 30)
        blue_shift = random.randint(10, 25)
        dust_amount = random.randint(200, 600)
        scratch_amount = random.randint(5, 25)
        noise_amount = random.uniform(0.01, 0.05)
        fray_cuts = random.randint(150, 400)
        water_stain_amount = random.randint(3, 10)
        fold_lines_amount = random.randint(1, 5)
        color_cast_intensity = (10, 40)

    elif damage_level == "heavy":
        lightness_reduction = random.randint(30, 60)
        blue_shift = random.randint(20, 50)
        dust_amount = random.randint(600, 1200)
        scratch_amount = random.randint(20, 50)
        noise_amount = random.uniform(0.05, 0.15)
        fray_cuts = random.randint(400, 800)
        water_stain_amount = random.randint(10, 20)
        fold_lines_amount = random.randint(5, 10)
        color_cast_intensity = (30, 70)

    else:
        raise ValueError("damage_level must be 'normal' or 'heavy'")

    image = fade_colors(image, lightness_reduction, blue_shift)
    image = add_color_cast(image, intensity_range=color_cast_intensity)
    image = add_dust(image, dust_amount)
    image = add_scratches(image, scratch_amount)
    image = add_noise(image, noise_amount)
    image = fray_edges(image, fray_cuts)
    image = add_water_stains(image, water_stain_amount)
    image = add_fold_lines(image, fold_lines_amount)

    return image


def batch_process(input_folder, output_folder):
    """Batch process all images in a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"damaged_{filename}")

            image = cv2.imread(input_path)
            if image is None:
                print(f"Skipping {filename} (could not read)")
                continue

            damaged = simulate_scan_artifacts(image, "heavy")
            cv2.imwrite(output_path, damaged)
            print(f"Processed {filename} -> {output_path}")

# Example usage
batch_process("../data", "defects_added_images")
