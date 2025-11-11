# app.py
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from datetime import datetime
import os, json, cv2, re, io
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
from collections import defaultdict
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ------------------------------------------------
# 🧠 Advanced Text Parser
# ------------------------------------------------
class AttributeParser:
    """Parse text descriptions into structured attributes"""

    def __init__(self):
        self.age_keywords = {
            'young': ['young', 'teenage', 'youth', 'adolescent'],
            'middle-aged': ['middle-aged', 'adult', 'mature'],
            'elderly': ['elderly', 'old', 'senior', 'aged']
        }

        self.hair_colors = ['black', 'brown', 'blonde', 'red', 'gray', 'white', 'auburn']
        self.eye_colors = ['blue', 'green', 'brown', 'hazel', 'gray']
        self.hair_styles = ['short', 'long', 'curly', 'straight', 'wavy', 'braided']
        self.accessories = ['glasses', 'hat', 'earrings', 'necklace']
        self.expressions = ['smile', 'smiling', 'serious', 'cheerful', 'neutral', 'happy']

    def parse(self, text):
        """Extract structured attributes from text"""
        text_lower = text.lower()

        attributes = {
            'age_group': None,
            'gender': None,
            'hair_color': None,
            'hair_style': None,
            'eye_color': None,
            'accessories': [],
            'expression': None,
            'full_text': text
        }

        # Age detection
        for age_cat, keywords in self.age_keywords.items():
            if any(kw in text_lower for kw in keywords):
                attributes['age_group'] = age_cat
                break

        # Gender detection
        if any(word in text_lower for word in ['man', 'male', 'boy', 'gentleman']):
            attributes['gender'] = 'male'
        elif any(word in text_lower for word in ['woman', 'female', 'girl', 'lady']):
            attributes['gender'] = 'female'

        # Hair attributes
        for color in self.hair_colors:
            if color in text_lower:
                attributes['hair_color'] = color
                break

        for style in self.hair_styles:
            if style in text_lower and 'hair' in text_lower:
                attributes['hair_style'] = style
                break

        # Eye color
        for color in self.eye_colors:
            if color in text_lower and 'eye' in text_lower:
                attributes['eye_color'] = color
                break

        # Accessories
        attributes['accessories'] = [acc for acc in self.accessories if acc in text_lower]

        # Expression
        for expr in self.expressions:
            if expr in text_lower:
                attributes['expression'] = expr
                break

        return attributes

# ------------------------------------------------
# 🧠 Enhanced Generator with Face Analysis
# ------------------------------------------------
class AdvancedTextToFaceGenerator:
    def __init__(self):
        """Initialize with advanced face analysis capabilities"""
        
        # Use Streamlit caching for model loading
        @st.cache_resource
        def load_models():
            print("🔄 Loading models (this may take a minute)...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Using device: {device}")

            # Load CLIP
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # Load Stable Diffusion
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None
            ).to(device)

            if torch.cuda.is_available():
                pipe.enable_attention_slicing()

            return clip_model, clip_processor, pipe, device

        self.clip_model, self.clip_processor, self.pipe, self.device = load_models()
        
        # Initialize parser
        self.parser = AttributeParser()

        # Face detector (OpenCV Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        print("✅ Models loaded successfully!\n")

    # ------------------------------------------------
    def generate_face(self, text_description, num_inference_steps=50, guidance_scale=7.5, seed=None):
        """Generate face with optional seed for reproducibility"""

        enhanced_prompt = (
            f"high quality portrait photograph of {text_description}, "
            "professional headshot, detailed facial features, realistic lighting, "
            "sharp focus, 4k, centered composition"
        )

        # Set seed if provided
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        start = datetime.now()

        with torch.inference_mode():
            image = self.pipe(
                enhanced_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
                generator=generator
            ).images[0]

        gen_time = (datetime.now() - start).total_seconds()
        return image, gen_time

    # ------------------------------------------------
    def calculate_clip_similarity(self, text, image):
        """Calculate CLIP similarity with proper normalization"""

        # Process inputs
        inputs = self.clip_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get embeddings
            vision_outputs = self.clip_model.vision_model(**{k: v for k, v in inputs.items() if k == 'pixel_values'})
            text_outputs = self.clip_model.text_model(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})

            # Get pooled outputs
            image_embeds = self.clip_model.visual_projection(vision_outputs.pooler_output)
            text_embeds = self.clip_model.text_projection(text_outputs.pooler_output)

            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Compute cosine similarity
            similarity = torch.cosine_similarity(image_embeds, text_embeds).item()

            # Convert from [-1, 1] to [0, 1]
            normalized_similarity = (similarity + 1) / 2

        return normalized_similarity

    # ------------------------------------------------
    def calculate_attribute_specific_scores(self, attributes, image):
        """Calculate CLIP scores for individual attributes"""

        scores = {}

        if attributes['hair_color']:
            prompt = f"a person with {attributes['hair_color']} hair"
            scores['hair_color'] = self.calculate_clip_similarity(prompt, image)

        if attributes['eye_color']:
            prompt = f"a person with {attributes['eye_color']} eyes"
            scores['eye_color'] = self.calculate_clip_similarity(prompt, image)

        if attributes['age_group']:
            prompt = f"a {attributes['age_group']} person"
            scores['age_group'] = self.calculate_clip_similarity(prompt, image)

        if attributes['expression']:
            prompt = f"a person with a {attributes['expression']} expression"
            scores['expression'] = self.calculate_clip_similarity(prompt, image)

        return scores

    # ------------------------------------------------
    # In your AdvancedTextToFaceGenerator class, replace the mediapipe part:

def detect_faces(self, image):
    """Detect faces using OpenCV only (no mediapipe)"""
    img_array = np.array(image)
    
    # Convert to grayscale for face detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Use OpenCV's built-in face detector
    faces = self.face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    analysis = {
        'num_faces': len(faces),
        'face_detected': len(faces) > 0,
        'face_size_ratio': 0.0,
        'face_centered': False
    }
    
    if len(faces) > 0:
        # Get largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Calculate face size ratio
        img_area = img_array.shape[0] * img_array.shape[1]
        face_area = w * h
        analysis['face_size_ratio'] = face_area / img_area
        
        # Check if centered
        center_x = x + w/2
        center_y = y + h/2
        img_center_x = img_array.shape[1] / 2
        img_center_y = img_array.shape[0] / 2
        
        distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
        threshold = img_array.shape[1] * 0.2
        analysis['face_centered'] = distance < threshold
    
    return analysis
    # ------------------------------------------------
    def evaluate_face_quality(self, image):
        """Comprehensive quality metrics"""

        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Contrast (standard deviation)
        contrast = gray.std()

        # Brightness
        brightness = gray.mean()

        # Color diversity (RGB variance)
        color_variance = np.var(img_array, axis=(0,1)).mean()

        # Edge density (Canny edge detection)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        return {
            "sharpness": float(sharpness),
            "contrast": float(contrast),
            "brightness": float(brightness),
            "color_variance": float(color_variance),
            "edge_density": float(edge_density)
        }

# ------------------------------------------------
# 📊 Streamlit UI Components
# ------------------------------------------------
def create_single_generation_tab(generator):
    st.header("🎨 Single Face Generation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        text_input = st.text_area(
            "📝 Face Description",
            value="a young woman with long brown hair and green eyes",
            height=100,
            placeholder="e.g., a young man with short black hair and glasses"
        )
        
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            steps = st.slider("Inference Steps", 20, 100, 50)
        with col1b:
            guidance = st.slider("Guidance Scale", 5.0, 15.0, 7.5, 0.5)
        with col1c:
            seed = st.text_input("🎲 Seed", value="", placeholder="Random if empty")
        
        generate_btn = st.button("🚀 Generate Face", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### Generated Face")
        image_placeholder = st.empty()
        info_placeholder = st.empty()
    
    # Example prompts
    with st.expander("💡 Example Prompts"):
        examples = [
            "a young man with short black hair and glasses",
            "an elderly woman with gray hair and a warm smile", 
            "a teenager with curly red hair and freckles",
            "a middle-aged man with a beard and brown eyes"
        ]
        for example in examples:
            if st.button(example, key=example, use_container_width=True):
                st.session_state.example_text = example
                st.rerun()
    
    if generate_btn and text_input.strip():
        with st.spinner("🔄 Generating face... This may take 15-30 seconds"):
            try:
                seed_val = int(seed) if seed.strip() else None
                image, gen_time = generator.generate_face(text_input, steps, guidance, seed_val)
                
                # Display image
                image_placeholder.image(image, use_column_width=True)
                
                # Comprehensive analysis
                with st.spinner("🔍 Analyzing results..."):
                    clip_score = generator.calculate_clip_similarity(text_input, image)
                    attributes = generator.parser.parse(text_input)
                    attr_scores = generator.calculate_attribute_specific_scores(attributes, image)
                    quality = generator.evaluate_face_quality(image)
                    face_analysis = generator.detect_faces(image)
                
                # Display metrics
                with info_placeholder.container():
                    st.success("✅ Generation Complete!")
                    
                    # Metrics in columns
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    
                    with col_metrics1:
                        st.metric("⏱️ Generation Time", f"{gen_time:.2f}s")
                        st.metric("🎯 CLIP Score", f"{clip_score:.3f}")
                    
                    with col_metrics2:
                        st.metric("👤 Face Detected", "✅ Yes" if face_analysis['face_detected'] else "❌ No")
                        st.metric("📐 Face Coverage", f"{face_analysis['face_size_ratio']*100:.1f}%")
                    
                    with col_metrics3:
                        st.metric("🎯 Centered", "✅" if face_analysis['face_centered'] else "❌")
                        st.metric("🔢 Seed", seed_val if seed_val else "Random")
                    
                    # Attributes section
                    st.subheader("🏷️ Detected Attributes")
                    attr_col1, attr_col2, attr_col3, attr_col4 = st.columns(4)
                    
                    with attr_col1:
                        st.write("**Age:**", attributes['age_group'] or "N/A")
                        st.write("**Gender:**", attributes['gender'] or "N/A")
                    
                    with attr_col2:
                        st.write("**Hair Color:**", attributes['hair_color'] or "N/A")
                        st.write("**Hair Style:**", attributes['hair_style'] or "N/A")
                    
                    with attr_col3:
                        st.write("**Eye Color:**", attributes['eye_color'] or "N/A")
                        st.write("**Expression:**", attributes['expression'] or "N/A")
                    
                    with attr_col4:
                        st.write("**Accessories:**", ", ".join(attributes['accessories']) or "None")
                    
                    # Attribute scores
                    if attr_scores:
                        st.subheader("⭐ Attribute Scores")
                        score_cols = st.columns(len(attr_scores))
                        for idx, (attr, score) in enumerate(attr_scores.items()):
                            with score_cols[idx]:
                                st.metric(
                                    f"{attr.replace('_', ' ').title()}",
                                    f"{score:.3f}"
                                )
                    
                    # Quality metrics
                    st.subheader("🎨 Quality Metrics")
                    qual_col1, qual_col2, qual_col3, qual_col4, qual_col5 = st.columns(5)
                    quality_metrics = [
                        ("Sharpness", quality['sharpness'], "{:.1f}"),
                        ("Contrast", quality['contrast'], "{:.1f}"), 
                        ("Brightness", quality['brightness'], "{:.1f}"),
                        ("Color Var", quality['color_variance'], "{:.1f}"),
                        ("Edge Density", quality['edge_density'], "{:.3f}")
                    ]
                    
                    for idx, (name, value, fmt) in enumerate(quality_metrics):
                        with [qual_col1, qual_col2, qual_col3, qual_col4, qual_col5][idx]:
                            st.metric(name, fmt.format(value))
                
                # Download button
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    "💾 Download Image",
                    buf.getvalue(),
                    file_name=f"generated_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def create_batch_generation_tab(generator):
    st.header("📦 Batch Generation & Analysis")
    
    st.text_area(
        "Enter descriptions (one per line)",
        key="batch_descriptions",
        height=150,
        placeholder="a young man with short black hair\nan elderly woman with gray hair\na teenager with curly red hair"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        num_variations = st.slider("Variations per description", 1, 5, 3)
    with col2:
        batch_size = st.slider("Batch size (images per run)", 1, 10, 3)
    
    if st.button("🚀 Run Batch Generation", type="primary", use_container_width=True):
        descriptions = [d.strip() for d in st.session_state.batch_descriptions.split('\n') if d.strip()]
        
        if not descriptions:
            st.warning("⚠️ Please enter at least one description")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        total_images = len(descriptions) * num_variations
        
        for desc_idx, desc in enumerate(descriptions):
            status_text.text(f"Processing: '{desc}'")
            
            for var_idx in range(num_variations):
                progress = (desc_idx * num_variations + var_idx) / total_images
                progress_bar.progress(progress)
                
                try:
                    seed = desc_idx * 1000 + var_idx
                    image, gen_time = generator.generate_face(desc, seed=seed)
                    
                    # Calculate metrics
                    clip_score = generator.calculate_clip_similarity(desc, image)
                    attributes = generator.parser.parse(desc)
                    attr_scores = generator.calculate_attribute_specific_scores(attributes, image)
                    quality = generator.evaluate_face_quality(image)
                    face_analysis = generator.detect_faces(image)
                    
                    # Convert image to bytes for display
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    
                    result = {
                        "description": desc,
                        "variation": var_idx + 1,
                        "image": buf.getvalue(),
                        "clip_score": clip_score,
                        "generation_time": gen_time,
                        "face_detected": face_analysis['face_detected'],
                        "face_coverage": face_analysis['face_size_ratio'] * 100,
                        "seed": seed,
                        **attributes
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    st.error(f"Error generating {desc} variation {var_idx+1}: {str(e)}")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Batch generation complete!")
        
        # Display results
        st.subheader("📊 Batch Results")
        
        # Summary statistics
        if all_results:
            df = pd.DataFrame(all_results)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", len(df))
            with col2:
                st.metric("Avg CLIP Score", f"{df['clip_score'].mean():.3f}")
            with col3:
                st.metric("Face Detection Rate", f"{(df['face_detected'].sum()/len(df)*100):.1f}%")
            with col4:
                st.metric("Avg Generation Time", f"{df['generation_time'].mean():.2f}s")
            
            # Display images in a grid
            st.subheader("🖼️ Generated Faces")
            for i in range(0, len(all_results), 3):
                cols = st.columns(3)
                for j in range(3):
                    idx = i + j
                    if idx < len(all_results):
                        with cols[j]:
                            result = all_results[idx]
                            st.image(result['image'], use_column_width=True)
                            st.caption(f"**{result['description'][:30]}...**")
                            st.caption(f"CLIP: {result['clip_score']:.3f} | "
                                     f"Face: {'✅' if result['face_detected'] else '❌'}")
            
            # Download results as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def main():
    st.set_page_config(
        page_title="Advanced Text-to-Face Generator",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #6a0dad;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #6a0dad;
        background-color: #f8f9fa;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧬 Advanced Text-to-Face Generator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
    <strong>Multi-attribute parsing • Face detection • Comprehensive analysis • Batch generation</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Features overview
    with st.expander("🚀 Features Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-card">
            <h4>🎨 Multi-Attribute Parsing</h4>
            <p>Extracts age, gender, hair, eyes, accessories from text</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-card">
            <h4>🔍 Face Analysis</h4>
            <p>Detects faces, analyzes positioning and quality metrics</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-card">
            <h4>📊 Advanced Metrics</h4>
            <p>CLIP similarity, sharpness, contrast, edge density</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Initialize generator (cached)
    if 'generator' not in st.session_state:
        with st.spinner("🔄 Loading AI models... This may take a few minutes"):
            st.session_state.generator = AdvancedTextToFaceGenerator()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎨 Single Generation", "📦 Batch Generation"])
    
    with tab1:
        create_single_generation_tab(st.session_state.generator)
    
    with tab2:
        create_batch_generation_tab(st.session_state.generator)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit • Powered by Stable Diffusion & CLIP • "
        "Text-to-Face Generation System"
    )

if __name__ == "__main__":

    main()
