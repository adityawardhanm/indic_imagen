import os 
import streamlit as st
import pandas as pd
from synimagen import synimagen , tempfilemanager
import random
import time
from threading import Thread
from queue import Queue
import numpy as np
import re


# Initialize generator
gen = synimagen()

#================================================================================================
#Streamlit Configuration
#================================================================================================

def count_eligible_lines(file_path, max_length):
    """First pass to count eligible lines with progress"""
    total = 0
    eligible = 0
    
    with st.status("📊 Scanning file...", expanded=True) as status:
        st.write("Counting total lines and eligible entries...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total += 1
                if len(line.strip()) <= max_length:
                    eligible += 1
                
                # Update progress every 1000 lines
                if i % 1000 == 0:
                    progress = i / (i + 1)  # Fake progress until we know total
                    progress_bar.progress(min(progress, 0.99))
                    status_text.text(f"Lines processed: {i}")

        progress_bar.progress(1.0)
        status_text.text(f"Complete! Total lines: {total}, Eligible lines: {eligible}")
        status.update(label="File scan complete ✅", state="complete")
    
    return total, eligible

def optimized_sample(file_path, max_length, target_size):
    """Single-pass sampling with guaranteed target size"""
    eligible_lines = []
    buffer_size = 1024 * 1024  # 1MB chunks
    file_size = os.path.getsize(file_path)
    leftover = ''  # Track incomplete lines between buffers

    with st.status("⚡ Turbo Sampling...", expanded=True) as status:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        bytes_processed = 0

        # First pass: collect all eligible lines
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                buffer = f.read(buffer_size)
                if not buffer:  # End of file
                    break
                bytes_processed += len(buffer)
                
                # Combine with leftover and split lines
                buffer = leftover + buffer
                lines = buffer.split('\n')
                leftover = lines.pop()  # Save incomplete line
                
                # Process complete lines
                for line in lines:
                    clean = line.strip()
                    if clean and len(clean) <= max_length:
                        eligible_lines.append(clean)
                
                # Update progress
                progress = bytes_processed / file_size
                progress_bar.progress(progress)
                status_text.text(f"Processed: {bytes_processed/1e9:.2f}GB | Eligible lines: {len(eligible_lines)}")

        # Process final leftover
        if leftover:
            clean = leftover.strip()
            if clean and len(clean) <= max_length:
                eligible_lines.append(clean)

        # Second pass: take exact sample (removed nested status)
        sampling_text = st.empty()
        sampling_text.text(f"Found {len(eligible_lines)} eligible lines, sampling {target_size}")
        
        if len(eligible_lines) <= target_size:
            sample = eligible_lines
            sampling_text.text(f"Using all {len(sample)} available lines (fewer than requested {target_size})")
        else:
            rng = np.random.default_rng()
            sample_indices = rng.choice(len(eligible_lines), size=target_size, replace=False)
            sample = [eligible_lines[i] for i in sample_indices]
            sampling_text.text(f"Sampled exactly {len(sample)} lines")

        # Finalize
        progress_bar.progress(1.0)
        status.update(label=f"Sampling complete: got {len(sample)} lines ⚡", state="complete")
    
    return pd.DataFrame(sample, columns=["text"])

# Define the Streamlit config directory
config_dir = os.path.expanduser("~/.streamlit")
config_path = os.path.join(config_dir, "config.toml")

# Ensure the directory exists
os.makedirs(config_dir, exist_ok=True)

# Automatically create config.toml if it doesn't exist
if not os.path.exists(config_path):
    with open(config_path, "w") as config_file:
        config_file.write("[server]\nmaxUploadSize = 25000\n")  # 25 GB
else:
    # Update existing config.toml if needed
    with open(config_path, "r+") as config_file:
        content = config_file.read()
        if "maxUploadSize" not in content:
            config_file.write("\n[server]\nmaxUploadSize = 25000\n")
#================================================================================================
#Global Configuration
#================================================================================================

num_threads_available = os.cpu_count()-2
class_options = ["Class 1", "Class 2", "Class 3"]
num_threads = 1

#================================================================================================
#Streamlit App
#================================================================================================


def main():
    # Title 
    st.set_page_config(page_title="Synthetic Image Generator", 
                        layout="centered",                   
                        page_icon="🧊",
)
    
    # About Synthetic Image Generator
    st.title("Synthetic Image Generator")
    
    st.write("This application allows you to generate high-quality synthetic images from text data with extensive customization options.")
    
    st.markdown("### 🔍 About the App")
    st.write("This tool is a practical implementation of the paper: [OCR Synthetic Benchmark Dataset for Indic Languages](https://arxiv.org/abs/2205.02543). While maintaining the core principles, some modifications have been introduced for enhanced usability.")
    
    st.markdown("## 🏷️ Class Descriptions")
    st.write("**Class 1:** Text with a white background. Choose between the base paper color configuration or WCAG 2.0 compliant 90 color pairs.")
    st.write("**Class 2:** Text with a random background. Adjustable noise levels and color grading using base paper settings or WCAG 2.0 guidelines.")
    st.write("**Class 3:** Poor-quality scanned text documents. Adjustable noise levels and color grading using base paper settings or WCAG 2.0 guidelines.")
    
    st.markdown("## 🎛️ Customization Options")
    st.write("🔹 Adjust noise levels to simulate real-world scanning conditions.")
    st.write("🔹 Choose from different types of noise to fine-tune the synthetic images.")
    st.write("🔹 Generate high-quality datasets tailored for OCR model training.")
    
    st.markdown("## 📂 Output Details")
    st.write("The generated image dataset, along with the output mapping, is stored in the same directory as this program.")
    st.write("For the best results, it is recommended to use Unicode fonts.")
    
    st.markdown("---")

    # Step 1: Global configuration
    st.markdown("###  Global Config")
    file = st.text_input("Give file path:")
    font = st.file_uploader("Upload a font file", type=["ttf", "otf"], accept_multiple_files=True)
    max_character_length = st.number_input("Max character length", min_value=1, max_value=1000, value=100)
    height = st.number_input("Height range (in px)", min_value=32, max_value=1000, value=100)
    dpi = st.number_input("DPI", min_value=1, max_value=1000, value=90)
    max_storage_size = st.slider("Max storage size (in Gbs)", min_value=1, max_value=10, value=1) * 1024**3


    multi_threading = st.checkbox("Enable multi-threading", value=False)
    if multi_threading:
        num_threads = st.number_input("Number of threads", min_value=1, max_value=num_threads_available, value=num_threads_available-(num_threads_available//2))
    
    # Step 2: Class Configuration
    select_all = st.checkbox("Select All Classes?", value=True)
    selected_classes = class_options if select_all else st.multiselect("Choose Classes to Select: ", class_options, default=class_options)

    # Step 2.1: Dynamically set ratios only for selected classes
    ratios = {}

    if len(selected_classes) == 1:
        ratios = {selected_classes[0]: 100}                         # If only one class is selected, it gets 100%
        
    elif len(selected_classes) == 2:
        class1, class2 = selected_classes                           # If two classes are selected, use a single slider (0-100%) and calculate the other
        value = st.slider(f"Adjust {class1} (%)", 0, 100, 50)
        ratios = {class1: value, class2: 100 - value}

    elif len(selected_classes) == 3:
        class1, class2, class3 = selected_classes                   # If three classes are selected, use a two-handle slider
        val1, val2 = st.slider(
            "Adjust distribution (%)", 
            0, 100, (33, 66)  # Two handles with default positions
        )
        ratios = {
            class1: val1,
            class2: val2 - val1,
            class3: 100 - val2
        }

    # Step 2.2: Set Class-wise Configuration
    st.markdown("## 🎨 Class-wise Configuration")
    class_config = {}
    bg_images = st.file_uploader("Upload background images", type=["jpg", "png"], accept_multiple_files=True) if "Class 2" or "Class 3" in selected_classes else None
    
    for cls, ratio in ratios.items():
        with st.container():
            st.markdown(f"### 🎯 {cls}: {ratio}%" )
            color_config = st.radio(f"Color Configuration ({cls})", ["Base Paper", "WCAG 2.0 Compliant"], key=f"color_{cls}")
            
            class_entry = {
                "ratio": ratio/100,
                "color_config": color_config if cls in selected_classes else "Base Paper"
            }
            
            if cls == "Class 3":

                noise_min, noise_max, = st.slider(
                    "Adjust noise (%)", 
                    0.1, 0.9, (0.2, 0.4))  # Two handles with default positions  
                class_entry["noise_range"] = (noise_min, noise_max)
                noise_types = st.multiselect(
                    f"Noise Types ({cls})", 
                    ["gaussian", "salt_pepper", "speckle", "poisson"], 
                    default=["gaussian", "salt_pepper", "speckle"],
                    key=f"noise_types_{cls}"
                )
                class_entry["noise_types"] = noise_types
                blur_min, blur_max, = st.slider(
                    "Adjust blur (%)", 
                    0.1, 0.9, (0.2, 0.5))  # Two handles with default positions  
                class_entry["blur_range"] = (blur_min, blur_max)
            
            class_config[cls] = class_entry
    # Display the configuration in the required format
    st.markdown("# Configuration")
    st.code(str(class_config), language="python")
    file_manager_1 = tempfilemanager()
    file_manager_2 = tempfilemanager()
    font_paths = file_manager_1.load_files(font)
    bg_images = file_manager_2.load_files(bg_images) if bg_images else None
    if st.button("Generate Images"):
        if file:
            # File scanning phase
            total_lines, eligible_lines = count_eligible_lines(file, max_character_length)
            target_size = int(eligible_lines*0.0027)
            print(target_size)
            df = optimized_sample(file, max_character_length, target_size)
            
            # Image generation setup
            progress_queue = Queue()
            st.session_state.generation_progress = 0
            total_images = len(df)
            # Start generation in background thread
            def run_generation():
                gen.generate_images(
                    df=df,
                    font_paths=font_paths,
                    bg_images=bg_images,
                    dpi=dpi,
                    height=height,
                    max_char_length=max_character_length,
                    max_storage_size=max_storage_size,
                    num_threads=num_threads,
                    class_config=class_config,
                    progress_queue=progress_queue
                )

            thread = Thread(target=run_generation)
            thread.start()
            
            # Progress monitoring
            with st.status("🎨 Generating images...", expanded=True) as status:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                start_time = time.time()
                
                while thread.is_alive():
                    # Update from queue
                    while not progress_queue.empty():
                        st.session_state.generation_progress += progress_queue.get()
                    
                    # Calculate metrics
                    elapsed = time.time() - start_time
                    processed = st.session_state.generation_progress
                    progress = processed / total_images
                    images_sec = processed / elapsed if elapsed > 0 else 0
                    
                    # Update display
                    progress_bar.progress(progress)
                    status_text.markdown(f"""
                        **Progress:** {processed}/{total_images} images  
                        **Speed:** {images_sec:.1f} images/sec  
                        **Elapsed:** {time.strftime('%H:%M:%S', time.gmtime(elapsed))}
                    """)
                    
                    time.sleep(0.1)
                
                progress_bar.progress(1.0)
                status.update(label="Generation complete ✅", state="complete")

            st.success(f"Successfully generated {total_images} images!")
            file_manager_1.cleanup()
            file_manager_2.cleanup()
        else:
            st.error("Please upload a text file")

if __name__ == "__main__":
    main()
