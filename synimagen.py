import os
import random
import tempfile
import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageFont, ImageDraw, ImageOps, ImageEnhance
import psutil
import csv
import traceback

#================================================================================================
# Global Configuration
#================================================================================================

# WCAG 2.0 contrast ratio (in standard AA Compliance Level, i.e., 4.5:1)
class wcag20:
    def __init__(self, ratio=4.5):
        self.ratio = ratio
    @staticmethod
    def rel_lum(rgb):
        s = [(x/255)/12.92 if (x/255) <= 0.03928 else ((x/255 + 0.055)/1.055) ** 2.4 for x in rgb]
        return s[0] * 0.2126 + s[1] * 0.7152 + s[2] * 0.0722

    def contrast(self, c1, c2):
        l1, l2 = self.rel_lum(c1), self.rel_lum(c2)
        return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

    def valid_color_pairs(self, bg, txt):
        return [(b, t) for b in bg for t in txt if b != t and self.contrast(b, t) >= self.ratio]

#================================================================================================
#Color-Config Variables
#================================================================================================

wcag_colors = [
    (255, 255, 255), (0, 0, 0), (128, 128, 128), (255, 0, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (33, 150, 243),
    (76, 175, 80), (244, 67, 54), (156, 39, 176), (255, 152, 0), (66, 103, 178),
    (29, 161, 242), (225, 48, 108), (0, 119, 181), (255, 87, 51), (64, 224, 208),
    (255, 218, 185), (123, 104, 238), (50, 205, 50), (198, 40, 40), (46, 139, 87),
    (218, 112, 214), (210, 105, 30), (255, 140, 0), (245, 245, 245), (169, 169, 169)
]


pairs_base_colors = [
    ((255, 255, 255), (0, 0, 0)), ((250, 250, 250), (20, 20, 20)), ((245, 245, 245), (30, 30, 30)), 
    ((240, 240, 240), (50, 50, 50)), ((235, 235, 235), (60, 30, 30)), ((230, 230, 230), (25, 50, 75)), 
    ((225, 225, 225), (90, 0, 0)), ((220, 220, 220), (0, 70, 0)), ((215, 215, 215), (10, 10, 80)), 
    ((210, 210, 210), (80, 40, 0)), ((250, 250, 245), (0, 0, 80)), ((245, 245, 240), (70, 0, 100)), 
    ((240, 240, 235), (128, 0, 0)), ((235, 235, 230), (0, 50, 100)), ((230, 230, 225), (80, 20, 60)), 
    ((225, 225, 220), (128, 50, 0)), ((220, 220, 215), (100, 0, 80)), ((215, 215, 210), (80, 20, 20)), 
    ((210, 210, 205), (0, 100, 100)), ((205, 205, 200), (50, 50, 50)), ((255, 253, 250), (0, 40, 90)), 
    ((250, 248, 245), (40, 0, 60)), ((245, 243, 240), (90, 30, 0)), ((240, 238, 235), (100, 0, 0)), 
    ((235, 233, 230), (10, 60, 90)), ((230, 228, 225), (80, 40, 0)), ((225, 223, 220), (0, 80, 0)), 
    ((220, 218, 215), (60, 10, 10)), ((215, 213, 210), (50, 20, 80)), ((210, 208, 205), (0, 0, 90))
]

pairs_wcag_colors = wcag20().valid_color_pairs(wcag_colors, wcag_colors)

#================================================================================================
# Class/Functions Definitions
#================================================================================================



class tempfilemanager:
    def __init__(self):
        """Initialize with a new temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_paths = []

    def load_files(self, uploaded_files):
        """Save uploaded files to the temp directory and store paths."""
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(self.temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                self.file_paths.append(file_path)

        return self.file_paths

    def cleanup(self):
        """Delete all files and remove the temporary directory if empty."""
        if not self.file_paths:
            return  # Nothing to clean up

        for file_path in self.file_paths:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)  # Delete each file
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")

        # Remove directory if empty
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass  # Directory not empty, ignore

        # Reset paths after cleanup
        self.file_paths = []

def apply_noise(image: Image.Image, noise_range: tuple, noise_types: list): 
    noise_type = random.choice(noise_types)
    img_array = np.array(image)
    
    if noise_type == "poisson":
        vals = 2 ** np.ceil(np.log2(np.max(img_array)))
        noisy_img = np.random.poisson(img_array * vals) / float(vals)
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5  # Ratio of salt to pepper
        amount = random.uniform(*noise_range)
        noisy_img = np.copy(img_array)
        num_salt = np.ceil(amount * img_array.size * s_vs_p)
        num_pepper = np.ceil(amount * img_array.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
        noisy_img[tuple(coords)] = 255
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
        noisy_img[tuple(coords)] = 0
    
    elif noise_type == "gaussian":
        base_noise = random.uniform(*noise_range)
        adjusted_noise = base_noise * 255
        noise = np.random.normal(0, adjusted_noise, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    elif noise_type == "speckle":
        base_noise = random.uniform(*noise_range)
        noise = np.random.normal(0, base_noise, img_array.shape)
        noisy_img = img_array + img_array * noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_img)

def apply_blur(image: Image.Image, blur_level: tuple):
    base_blur = random.uniform(*blur_level)
    if base_blur <= 0:
        return image
    
    image = image.filter(ImageFilter.GaussianBlur(base_blur))
    
    return image

def apply_color(color, image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    img_array = np.array(image, dtype=np.float32)
    avg_color = img_array.mean(axis=(0, 1))
    shift_ratio = np.array(color, dtype=np.float32) / avg_color
    graded_array = np.clip(img_array * shift_ratio, 0, 255).astype(np.uint8)
    return Image.fromarray(graded_array)

def random_crop(bg_image: Image.Image, width: int, height: int) -> Image.Image:
    if bg_image.width <= width or bg_image.height <= height:
        scale_factor = max(width/ bg_image.width, height / bg_image.height)
        new_w, new_h = int(bg_image.width * scale_factor)+1, int(bg_image.height * scale_factor)+1
        bg_image = bg_image.resize((new_w, new_h), Image.LANCZOS)
    max_x, max_y = bg_image.width - width, bg_image.height - height
    start_x, start_y = random.randint(0, max_x) if max_x > 0 else 0, random.randint(0, max_y) if max_y > 0 else 0
    return bg_image.crop((start_x, start_y, start_x + width, start_y + height))



#================================================================================================

class largefilesampler:
    def __init__(self, df, max_char_length, sample_fraction= 1, memory_fraction=0.1):
        self.df = df
        self.max_char_length = max_char_length
        self.chunk_size = self.get_dynamic_chunk_size(memory_fraction)
        self.num_workers = mp.cpu_count()
        self.sample_fraction = sample_fraction

    def get_dynamic_chunk_size(self, fraction=0.05):
        free_mem = psutil.virtual_memory().available  # Available RAM in bytes
        return max(int(free_mem * fraction / 100), 1000)  # At least 1000 rows per chunk

    def process_chunk(self, chunk):
        return chunk[chunk.iloc[:, 0].str.len() <= self.max_char_length]

    def sample_dataframe(self):
        chunks = [self.df[i:i+self.chunk_size] for i in range(0, len(self.df), self.chunk_size)]
        pool = mp.Pool(self.num_workers)
        results = pool.map(self.process_chunk, chunks)
        pool.close(), pool.join()
        filtered_df = pd.concat(results, ignore_index=True)
        sampled_df = filtered_df.sample(frac=self.sample_fraction, random_state=42)
        print(f"Original dataset: {len(self.df):,} rows → After filtering: {len(filtered_df):,} rows → Sampled: {len(sampled_df):,} rows")
        return sampled_df


#================================================================================================\
# Generate Images
#================================================================================================

class synimagen:
    def __init__(self, output_base_dir: str = "./generated_images"):
        """Initialize the Synthetic Image Generator"""
        self.output_base_dir = output_base_dir
        self.csv_path = "./image_mappings.csv"
        self.total_images_generated = 0
        self.progress_queue = None  # Will be set during generation
            
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_base_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['path', 'text', 'font', 'class'])

    def process_data(self, df: pd.DataFrame, max_char_length: int, class_ratios: Dict) -> Dict[str, pd.DataFrame]:
        """Process the input dataframe and split into class-specific dataframes"""
        sampler = largefilesampler(df, max_char_length=max_char_length)
        filtered_df = sampler.sample_dataframe()
        
        # Split dataframe according to ratios
        total_samples = len(filtered_df)
        classes = list(class_ratios.keys())
        
        # Calculate sizes for each class
        sizes = {}
        cumulative = 0
        for cls in classes:
            ratio = class_ratios[cls]
            size = int(total_samples * ratio)
            sizes[cls] = size
            cumulative += size
        
        # Handle remaining samples due to integer division
        remaining = total_samples - cumulative
        if remaining > 0:
            for i in range(remaining):
                cls = classes[i % len(classes)]
                sizes[cls] += 1
        
        # Split the shuffled DataFrame
        shuffled_df = filtered_df.sample(frac=1).reset_index(drop=True)
        class_dfs = {}
        start = 0
        for cls in classes:
            end = start + sizes[cls]
            class_dfs[cls] = shuffled_df[start:end]
            start = end
        
        return class_dfs

    def append_to_csv(self, image_path: str, text: str, font: str, class_name: str):
        """Append an entry to the mapping CSV file"""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([image_path, text, font, class_name])
        if self.progress_queue:
            self.progress_queue.put(1)  # Signal one image completed
    def generate_class1(self, class1_df: pd.DataFrame, font_paths: List[str], dpi: int, height: int, config: Dict):
        """Generate images for Class 1 with dynamic text scaling"""
        # Choose the color pair list based on the config
        color_pairs = pairs_wcag_colors if config['color_config'] == "WCAG 2.0 Compliant" else pairs_base_colors

        for idx, row in class1_df.iterrows():
            try:
                # Get text from the DataFrame
                text = str(row.iloc[0])
                
                # Randomly select a font
                font_path = random.choice(font_paths)
                
                # Create a dummy image to measure text dimensions
                dummy_img = Image.new('RGB', (1, 1))
                dummy_draw = ImageDraw.Draw(dummy_img)
                
                # Define target text height as 80% of the image height
                target_text_height = int(height * 0.8)
                
                # Use binary search to find the maximum font size that fits within target_text_height
                low, high = 1, height  # possible font sizes
                optimal_font_size = low
                while low <= high:
                    mid = (low + high) // 2
                    font = ImageFont.truetype(font_path, size=mid)
                    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
                    current_text_height = text_bbox[3] - text_bbox[1]
                    
                    if current_text_height <= target_text_height:
                        optimal_font_size = mid  # this size fits, try a larger one
                        low = mid + 1
                    else:
                        high = mid - 1
                
                # Load the font with the optimal size
                font = ImageFont.truetype(font_path, size=optimal_font_size)
                text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Calculate image width (text width plus a 10-pixel margin)
                image_width = text_width + 10
                
                # Create the image with the chosen background color
                color_pair = random.choice(color_pairs)
                image = Image.new('RGB', (image_width, height), color_pair[0])
                draw = ImageDraw.Draw(image)
                
                # Center the text on the image
                text_x = (image_width - text_width) // 2
                text_y = (height - text_height) // 2
                draw.text((text_x, text_y), text, font=font, fill=color_pair[1])
                
                # Save the image with specified DPI, quality, and optimization
                image_path = f"{self.output_base_dir}/class1/img_{idx}.jpeg"
                image.save(image_path, dpi=(dpi, dpi), quality=90, optimize=True)
                
                # Record the generated image in CSV
                self.append_to_csv(image_path, text, os.path.basename(font_path), "Class 1")
                
            except Exception as e:
                print(f"Error generating Class 1 image for index {idx}: {str(e)}")
                print(f"Exception Type: {type(e)}")
                print(traceback.format_exc())
                continue

    def generate_class2(self, class2_df: pd.DataFrame, font_paths: List[str], bg_images: List[str], 
                       dpi: int, height: int, config: Dict):
        """Generate images for Class 2"""
        color_pairs = pairs_wcag_colors if config['color_config'] == "WCAG 2.0 Compliant" else pairs_base_colors

        for idx, row in class2_df.iterrows():
            try:
                # Get text from dataframe
                text = str(row.iloc[0])
                
                # Randomly select font and create font object
                font_path = random.choice(font_paths)
                font = ImageFont.truetype(font_path)
                
                # Calculate text dimensions
                dummy_img = Image.new('RGB', (1, 1))
                dummy_draw = ImageDraw.Draw(dummy_img)
                target_text_height = int(height * 0.8)

                # Use binary search to find the maximum font size that fits within target_text_height
                low, high = 1, height  # possible font sizes
                optimal_font_size = low
                while low <= high:
                    mid = (low + high) // 2
                    font = ImageFont.truetype(font_path, size=mid)
                    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
                    current_text_height = text_bbox[3] - text_bbox[1]
                    
                    if current_text_height <= target_text_height:
                        optimal_font_size = mid  # this size fits, try a larger one
                        low = mid + 1
                    else:
                        high = mid - 1

                font = ImageFont.truetype(font_path, size=optimal_font_size)
                text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                image_width = text_width + 10
                color_pair = random.choice(color_pairs)
                bg_image = Image.open(random.choice(bg_images))
                
                # Apply color grading if WCAG compliant is selected
                if config['color_config'] == "WCAG 2.0 Compliant":
                    bg_color = color_pair[0]
                    bg_image = apply_color(color=bg_color, image=bg_image)
                
                # Crop background to match text dimensions
                bg_image = random_crop(bg_image=bg_image, width=image_width, height=height)
                draw = ImageDraw.Draw(bg_image)
                
                # Center the text on the image
                text_x = (image_width - text_width) // 2
                text_y = (height - text_height) // 2
                
                # Draw text (black text for base paper, random WCAG color for compliant)
                text_color = color_pair[1]
                draw.text((text_x, text_y), text, font=font, fill=text_color)
                
                # Save image
                image_path = f"{self.output_base_dir}/class2/img_{idx}.jpeg"
                bg_image.save(image_path, dpi=(dpi, dpi), quality=90)
                
                # Add entry to CSV
                self.append_to_csv(image_path, text, os.path.basename(font_path), "Class 2")
                
            except Exception as e:
                print(f"Error generating Class 2 image for index {idx}: {str(e)}")
                print(f"Exception Type: {type(e)}")
                print(traceback.format_exc())
                continue

    def generate_class3(self, class3_df: pd.DataFrame, font_paths: List[str], bg_images: List[str], 
                   dpi: int, height: int, config: Dict):
        """Generate images for Class 3"""
        color_pairs = pairs_wcag_colors if config['color_config'] == "WCAG 2.0 Compliant" else pairs_base_colors
        
        for idx, row in class3_df.iterrows():
            try:
                # Get text from dataframe
                text = str(row.iloc[0])
                
                # Randomly select font and create font object
                font_path = random.choice(font_paths)
                font = ImageFont.truetype(font_path)
                
                # Calculate text dimensions
                dummy_img = Image.new('RGB', (1, 1))
                dummy_draw = ImageDraw.Draw(dummy_img)
                target_text_height = int(height * 0.8)

                # Use binary search to find the maximum font size that fits within target_text_height
                low, high = 1, height  # possible font sizes
                optimal_font_size = low
                while low <= high:
                    mid = (low + high) // 2
                    font = ImageFont.truetype(font_path, size=mid)
                    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
                    current_text_height = text_bbox[3] - text_bbox[1]
                    
                    if current_text_height <= target_text_height:
                        optimal_font_size = mid  # this size fits, try a larger one
                        low = mid + 1
                    else:
                        high = mid - 1

                font = ImageFont.truetype(font_path, size=optimal_font_size)
                text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                image_width = text_width + 10
                color_pair = random.choice(color_pairs)
                bg_image = Image.open(random.choice(bg_images))
                
                # Apply color grading if WCAG compliant is selected
                if config['color_config'] == "WCAG 2.0 Compliant":
                    bg_color = color_pair[0]
                    bg_image = apply_color(color=bg_color, image=bg_image)
                
                # Crop background to match text dimensions
                bg_image = random_crop(bg_image=bg_image, width=image_width, height=height)
                draw = ImageDraw.Draw(bg_image)
                
                # Center the text on the image
                text_x = (image_width - text_width) // 2
                text_y = (height - text_height) // 2
                
                # Draw text (black text for base paper, random WCAG color for compliant)
                text_color = color_pair[1]
                draw.text((text_x, text_y), text, font=font, fill=text_color)
                
                # Apply noise effects
                noisy_image = apply_noise(
                    image=bg_image, 
                    noise_range=config['noise_range'], 
                    noise_types=config['noise_types']
                )
                
                # Apply blur effect
                final_image = apply_blur(blur_level= config['blur_range'], image= noisy_image)
                
                # Save image
                image_path = f"{self.output_base_dir}/class3/img_{idx}.jpeg"
                final_image.save(image_path, dpi=(dpi, dpi), quality=90)
                
                # Add entry to CSV
                self.append_to_csv(image_path, text, os.path.basename(font_path), "Class 3")
                
            except Exception as e:
                print(f"Error generating Class 3 image for index {idx}: {str(e)}")
                print(f"Exception Type: {type(e)}")
                print(traceback.format_exc())
                continue

    def generate_class_chunk(self, chunk_data: pd.DataFrame, class_type: str, 
                           font_paths: List[str], bg_images: List[str], 
                           dpi: int, height: int, config: Dict):
        """Generate images for a chunk of data based on class type"""
        try:
            if class_type == "Class 1":
                self.generate_class1(chunk_data, font_paths, dpi, height, config)
            elif class_type == "Class 2":
                self.generate_class2(chunk_data, font_paths, bg_images, dpi, height, config)
            elif class_type == "Class 3":
                self.generate_class3(chunk_data, font_paths, bg_images, dpi, height, config)
        except Exception as e:
            print(f"Error processing chunk for {class_type}: {str(e)}")
            raise

    def split_dataframe(self, df: pd.DataFrame, num_chunks: int) -> List[pd.DataFrame]:
        """Split a DataFrame into approximately equal chunks"""
        chunk_size = math.ceil(len(df) / num_chunks)
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    def generate_images(self, df: pd.DataFrame, font_paths: List[str], bg_images: List[str],
                   dpi: int, height: int, max_char_length: int, max_storage_size: int,
                   num_threads: int, class_config: Dict, progress_queue):
        """Main function to generate all images using thread pooling"""
        # Extract ratios and active classes
        self.progress_queue = progress_queue        
        class_ratios = {class_name: settings['ratio'] for class_name, settings in class_config.items()}
        active_classes = [cls for cls in class_ratios if class_ratios[cls] > 0]
        
        # Create directories for active classes
        for cls in active_classes:
            dir_name = f"class{cls.split()[-1]}"  # 'Class 1' → 'class1'
            os.makedirs(f"{self.output_base_dir}/{dir_name}", exist_ok=True)
        
        # Process and split data
        class_dfs = self.process_data(df, max_char_length, class_ratios)
        
        # Calculate thread distribution based on active classes
        total_ratio = sum(class_ratios.values())
        thread_distribution = {}
        for cls in active_classes:
            thread_distribution[cls] = max(1, round(num_threads * class_ratios[cls] / total_ratio))
        
        # Adjust thread distribution if total exceeds num_threads
        while sum(thread_distribution.values()) > num_threads:
            max_cls = max(thread_distribution, key=lambda k: thread_distribution[k])
            thread_distribution[max_cls] -= 1
        
        # Create work chunks for each class
        work_chunks = {}
        for cls in active_classes:
            num_chunks = thread_distribution.get(cls, 1)
            work_chunks[cls] = self.split_dataframe(class_dfs[cls], num_chunks)
        
        # Prepare work items for thread pool
        work_items = []
        for cls, chunks in work_chunks.items():
            for chunk in chunks:
                if len(chunk) == 0:
                    continue  # Skip empty chunks
                # Determine if bg_images are needed
                bg = bg_images if cls in ['Class 2', 'Class 3'] else []
                work_items.append((
                    chunk,
                    cls,
                    font_paths,
                    bg,
                    dpi,
                    height,
                    class_config[cls]
                ))
        
        # Execute work items using thread pool
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for work_item in work_items:
                future = executor.submit(self.generate_class_chunk, *work_item)
                futures.append(future)
            
            # Handle exceptions
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in thread execution: {str(e)}")
                    raise