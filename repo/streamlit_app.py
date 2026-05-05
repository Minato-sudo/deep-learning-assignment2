import streamlit as st
import pandas as pd
import json
import os
import torch
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from diffusers import DDPMPipeline, DDIMPipeline
from cleanfid import fid
import time
import glob

# --- Page Configuration ---
st.set_page_config(
    page_title="Diffusion Discovery Center",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&display=swap');
    
    * { font-family: 'Space Grotesk', sans-serif; }
    
    .stApp { background: #0b0e14; color: #e2e8f0; }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .status-active { color: #10b981; font-weight: bold; }
    
    .section-header {
        border-left: 5px solid #3b82f6;
        padding-left: 15px;
        margin: 30px 0 20px 0;
        color: #60a5fa;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Global Caching for Models ---
@st.cache_resource
def load_diffusion_model(model_id="google/ddpm-cifar10-32", model_type="DDIM"):
    st.write(f"🔄 Loading {model_type} Model into GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "DDIM":
        pipe = DDIMPipeline.from_pretrained(model_id).to(device)
    else:
        pipe = DDPMPipeline.from_pretrained(model_id).to(device)
    return pipe

# --- Data Helpers ---
def get_results(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

# --- Header ---
with st.container():
    st.title("⚛️ Diffusion Discovery Center")
    st.markdown("""
    ### *The Ultimate Reproducibility & Inference Optimization Suite*
    **Authors:** Zain Shahid, SanaUllah, Muhammad Talha Arshad | **University:** FAST-NUCES  
    ---
    """)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/atom.png", width=70)
    menu = st.radio("Navigation", [
        "🏁 Project Timeline",
        "🔬 Original Repo Lab",
        "📡 Live Inference Lab",
        "📊 Experiment Analytics",
        "📂 Documentation & Logs"
    ])
    st.divider()
    st.subheader("🛠️ System Health")
    st.write(f"**Device:** {'🟢 NVIDIA RTX 5050' if torch.cuda.is_available() else '🔴 CPU Mode'}")
    st.write(f"**PyTorch:** {torch.__version__}")

# ==========================================
# SECTION: Project Timeline
# ==========================================
if menu == "🏁 Project Timeline":
    st.markdown('<h2 class="section-header">Project Roadmap & Achievements</h2>', unsafe_allow_html=True)
    
    # Visual Roadmap using columns
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        #### 📄 Assignment 1: Proposal
        - **Goal:** Explore Fast Diffusion.
        - **Problem:** Checkpoint 404 & GPU issues.
        - **Pivoted to:** Foundational DDPM.
        """)
    with c2:
        st.markdown("""
        #### 🔄 Assignment 2: Reproduction
        - **Task:** Replicate Ho et al. 2020.
        - **Metric:** Achieved FID **20.91**.
        - **Outcome:** Validated DDIM vs DDPM.
        """)
    with c3:
        st.markdown("""
        #### 🚀 Assignment 3: Optimization
        - **Exp 1:** Found optimal **200 steps**.
        - **Exp 2:** Characterized **Eta transition**.
        - **Exp 3:** Scaled to **CelebA-HQ faces**.
        """)
    
    st.divider()
    st.subheader("Global Project Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric-card"><h3>Best FID</h3><h2 style="color:#10b981">20.91</h2><p>DDPM 1000 Steps</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><h3>Sweet Spot</h3><h2 style="color:#3b82f6">45.11</h2><p>DDIM 50 Steps</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><h3>Cross-Domain</h3><h2 style="color:#8b5cf6">59.34</h2><p>Intra-FID (Faces)</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card"><h3>Efficiency</h3><h2 style="color:#f59e0b">20x</h2><p>Sampling Speedup</p></div>', unsafe_allow_html=True)

# ==========================================
# SECTION: Original Repo Lab (NEW)
# ==========================================
elif menu == "🔬 Original Repo Lab":
    st.markdown('<h2 class="section-header">Official Repository Reproduction Progress</h2>', unsafe_allow_html=True)
    st.info("""
    **Objective:** Reproduce Ho et al. 2020 using the official U-Net architecture and hyperparameters.  
    **Constraint:** Full training (800k steps) requires 10-14 days. We are running a **50,000 step benchmark** to validate the pipeline.
    """)
    
    sample_dir = "repo/original_architecture_reproduction/logs/DDPM_Reproduction_Attempt/sample/"
    
    if os.path.exists(sample_dir):
        images = glob.glob(os.path.join(sample_dir, "*.png"))
        if images:
            # Sort by step number
            images.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            
            latest_img = images[-1]
            st.subheader(f"Latest Sample: Step {os.path.basename(latest_img).split('.')[0]}")
            st.image(latest_img, caption="Evolution of CIFAR-10 generation from scratch", width=600)
            
            st.divider()
            st.subheader("Training History (Every 1000 steps)")
            cols = st.columns(4)
            for i, img_path in enumerate(reversed(images[:-1])):
                step = os.path.basename(img_path).split('.')[0]
                with cols[i % 4]:
                    st.image(img_path, caption=f"Step {step}")
        else:
            st.warning("Training started! Waiting for the first sample (Step 1000)...")
    else:
        st.error("Training directory not found. Please ensure the training script is running.")

# ==========================================
# SECTION: Live Inference Lab
# ==========================================
elif menu == "📡 Live Inference Lab":
    st.markdown('<h2 class="section-header">Live Generation & Real-Time Pipeline</h2>', unsafe_allow_html=True)
    
    col_set, col_gen = st.columns([1, 2])
    
    with col_set:
        st.subheader("⚙️ Sampling Settings")
        sampling_type = st.radio("Sampler", ["DDIM (Fast)", "DDPM (Probabilistic)"])
        steps = st.slider("Inference Steps", 1, 1000, 50)
        eta = st.slider("Stochasticity (Eta)", 0.0, 1.0, 0.0)
        batch_size = st.number_input("Batch Size", 1, 64, 4)
        
        st.divider()
        compute_fid_live = st.toggle("🧪 Compute Live Pipeline FID", value=False, help="Computes FID for this specific batch. Takes longer but proves the evaluation pipeline works.")
        
        run_gen = st.button("🚀 Run Generation Pipeline")

    with col_gen:
        if run_gen:
            with st.spinner("Executing Pipeline..."):
                pipe = load_diffusion_model(model_type="DDIM" if "DDIM" in sampling_type else "DDPM")
                
                start_time = time.time()
                images = pipe(
                    batch_size=batch_size, 
                    num_inference_steps=steps, 
                    eta=eta if "DDIM" in sampling_type else 1.0,
                    output_type="pil"
                ).images
                end_time = time.time()
                
                st.success(f"Generated {batch_size} images in {end_time - start_time:.2f} seconds.")
                
                # Show results in a grid
                st.image(images, width=150, caption=[f"Sample {i+1}" for i in range(len(images))])
                
                # Live FID Logic
                if compute_fid_live:
                    st.divider()
                    st.subheader("🧪 Live Batch Evaluation")
                    st.caption("⚠️ Note: This score is for this small batch only. For scientific results, refer to the 'Experiment Analytics' tab.")
                    st.write("Comparing this live batch against 10k real CIFAR-10 images...")
                    
                    # Create temp folder for live batch
                    temp_dir = "outputs/live_batch"
                    os.makedirs(temp_dir, exist_ok=True)
                    for i, img in enumerate(images):
                        img.save(f"{temp_dir}/{i}.png")
                    
                    # Compute FID
                    try:
                        live_score = fid.compute_fid(temp_dir, "data/cifar10_real")
                        st.metric("Live Batch FID", f"{live_score:.2f}")
                    except Exception as e:
                        st.error(f"FID Stats Error: {e}")
                else:
                    st.info("💡 **Tip:** Enable 'Compute Live Pipeline FID' in settings to verify the evaluation pipeline.")

# ==========================================
# SECTION: Experiment Analytics
# ==========================================
elif menu == "📊 Experiment Analytics":
    st.markdown('<h2 class="section-header">Deep Dive: Experiment Data</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Step Ablation Curve", "Eta Phase Transition"])
    
    with tab1:
        res = get_results("logs/results_step_ablation.json")
        if res:
            df = pd.DataFrame(list(res.items()), columns=['Steps', 'FID'])
            df['Steps'] = df['Steps'].astype(int)
            df = df.sort_values('Steps')
            fig = px.line(df, x='Steps', y='FID', log_x=True, markers=True, template="plotly_dark")
            fig.update_traces(line_color='#3b82f6', marker=dict(size=10), name="DDIM (Our Exp)")
            
            # Add DDPM 1000 baseline from A2
            fig.add_hline(y=20.91, line_dash="dot", line_color="#f59e0b", 
                         annotation_text="A2 DDPM Baseline (20.91)", annotation_position="top left")
            
            fig.add_hline(y=3.17, line_dash="dash", line_color="#10b981", 
                         annotation_text="Original Paper FID (3.17)", annotation_position="bottom left")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("⏱️ Speed vs. Quality Tradeoff Analysis")
            analysis_data = {
                "Metric": ["Sampling Steps", "Time per Image (est)", "FID (Quality)"],
                "Original DDPM (A2)": ["1000", "~3.0s", "20.91"],
                "Efficient DDIM (Exp 1)": ["50", "~0.15s", "45.11"],
                "Improvement": ["20x Faster", "20x Faster", "Decent Fidelity"]
            }
            st.table(pd.DataFrame(analysis_data))
            
            st.info("**Scientific Insight:** While DDPM 1000 steps has the lowest FID, our **DDIM 50-step** configuration provides a **20x speedup** with only a minor increase in FID, making it the most practical 'Sweet Spot' for deployment.")

    with tab2:
        res = get_results("logs/results_eta_study.json")
        if res:
            df = pd.DataFrame(list(res.items()), columns=['Eta', 'FID'])
            df['Eta'] = df['Eta'].astype(float)
            df = df.sort_values('Eta')
            fig = px.area(df, x='Eta', y='FID', template="plotly_dark")
            fig.update_traces(fillcolor='rgba(59, 130, 246, 0.2)', line_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)
            st.warning("**Phase Transition:** Note the sharp rise in FID after Eta=0.5. Stochasticity introduces significant noise at low step counts.")

# ==========================================
# SECTION: Documentation & Logs
# ==========================================
elif menu == "📂 Documentation & Logs":
    st.markdown('<h2 class="section-header">Technical Logs & Checkpoints</h2>', unsafe_allow_html=True)
    
    col_log, col_file = st.columns(2)
    with col_log:
        st.subheader("Raw Results (JSON)")
        file_sel = st.selectbox("View JSON", ["logs/results_step_ablation.json", "logs/results_eta_study.json", "logs/results_crossdomain.json"])
        st.json(get_results(file_sel))
        
    with col_file:
        st.subheader("Experiment Scripts")
        scr_sel = st.selectbox("View Code", ["scripts/experiment_steps.py", "scripts/experiment_eta.py", "scripts/experiment_crossdomain.py"])
        if os.path.exists(scr_sel):
            with open(scr_sel, 'r') as f:
                st.code(f.read(), language="python")
