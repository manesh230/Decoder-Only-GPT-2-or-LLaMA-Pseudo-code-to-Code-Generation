"""
Pseudo-code to Code Generator - Streamlit App
Deployed on Streamlit Cloud
"""
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Page configuration
st.set_page_config(
    page_title="Pseudo-code to Code Generator",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
    .output-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the fine-tuned model (cached for performance)"""
    try:
        # For Streamlit Cloud, model should be in the repo or downloaded
        model_path = "models/gpt2_spoc_final/best_model"
        
        # Check if model exists locally
        if not os.path.exists(model_path):
            st.error("Model files not found! Please ensure model is uploaded to GitHub.")
            st.stop()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def generate_code(model, tokenizer, device, pseudo_code, temperature, max_length, num_samples):
    """Generate code from pseudo-code"""
    try:
        prompt = f"Pseudo: {pseudo_code} Code:"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2  # Reduce repetition
            )
        
        codes = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            
            if "Code:" in text:
                code = text.split("Code:")[1].strip()
            else:
                code = text
            
            # Clean up the output
            code = code.replace("Pseudo:", "").strip()
            codes.append(code)
        
        return codes
    
    except Exception as e:
        st.error(f"Error generating code: {e}")
        return []

def main():
    # Header
    st.title("💻 Pseudo-code to Code Generator")
    st.markdown("""
    ### Transform natural language descriptions into Python code using AI
    
    This app uses a **GPT-2 model fine-tuned with LoRA** on the SPoC dataset to generate 
    code from pseudo-code descriptions.
    """)
    
    st.markdown("---")
    
    # Load model
    with st.spinner("🔄 Loading model... (this may take a minute on first run)"):
        model, tokenizer, device = load_model()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.1,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Controls randomness. Lower = more focused, Higher = more creative"
        )
        
        max_length = st.slider(
            "📏 Max Length",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Maximum number of tokens to generate"
        )
        
        num_samples = st.slider(
            "🔢 Number of Outputs",
            min_value=1,
            max_value=3,
            value=1,
            help="Generate multiple different code samples"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Architecture**: GPT-2 + LoRA  
        **Training**: 20,000 samples  
        **Eval Loss**: 1.16  
        **Trainable Params**: 294,912  
        **Device**: {device.upper()}
        """)
        
        st.markdown("---")
        
        st.markdown("### 📚 About")
        st.markdown("""
        This model was fine-tuned using:
        - **Base Model**: GPT-2
        - **Method**: LoRA (Low-Rank Adaptation)
        - **Dataset**: SPoC (Search-based Pseudocode to Code)
        - **Metrics**: BLEU Score: 7.89
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📝 Input Pseudo-code")
        
        # Example selector
        st.markdown("**Quick Examples:**")
        examples = {
            "Custom input": "",
            "Add two numbers": "create a function that adds two numbers",
            "Print loop": "write a loop that prints numbers from 1 to 10",
            "Student class": "define a class for a student with name and age",
            "Reverse string": "create a function to reverse a string",
            "Binary search": "implement binary search algorithm",
            "Factorial": "write a recursive function to calculate factorial",
            "Sort list": "create a function to sort a list in ascending order",
            "Find maximum": "write a function to find the maximum value in a list"
        }
        
        example_choice = st.selectbox(
            "Select an example or write your own:",
            list(examples.keys()),
            label_visibility="collapsed"
        )
        
        default_value = examples[example_choice]
        
        pseudo_code = st.text_area(
            "Enter your pseudo-code description:",
            value=default_value,
            height=200,
            placeholder="Example: create a function that calculates the sum of all even numbers in a list",
            help="Describe what you want the code to do in plain English"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            generate_button = st.button(
                "🚀 Generate Code",
                type="primary",
                use_container_width=True
            )
        
        with col_btn2:
            clear_button = st.button(
                "🗑️ Clear",
                use_container_width=True
            )
        
        if clear_button:
            st.rerun()
    
    with col2:
        st.subheader("💻 Generated Code")
        
        if generate_button:
            if not pseudo_code.strip():
                st.warning("⚠️ Please enter some pseudo-code description!")
            else:
                with st.spinner("🔄 Generating code..."):
                    codes = generate_code(
                        model,
                        tokenizer,
                        device,
                        pseudo_code,
                        temperature,
                        max_length,
                        num_samples
                    )
                
                if codes:
                    for i, code in enumerate(codes, 1):
                        if num_samples > 1:
                            st.markdown(f"**Sample {i} of {num_samples}:**")
                        
                        # Display code
                        st.code(code, language="python", line_numbers=True)
                        
                        # Download button
                        col_dl1, col_dl2 = st.columns([3, 1])
                        with col_dl2:
                            st.download_button(
                                label="📥 Download",
                                data=code,
                                file_name=f"generated_code_{i}.py",
                                mime="text/plain",
                                key=f"download_{i}"
                            )
                        
                        if i < len(codes):
                            st.markdown("---")
                    
                    # Show statistics
                    avg_length = sum(len(c.split()) for c in codes) / len(codes)
                    st.success(f"✅ Generated {len(codes)} code sample(s) | Avg length: {avg_length:.0f} tokens")
                else:
                    st.error("❌ Failed to generate code. Please try again.")
        else:
            st.info("👈 Enter pseudo-code and click 'Generate Code' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using GPT-2 + LoRA | Fine-tuned on SPoC Dataset</p>
        <p>Streamlit • Transformers • PyTorch • PEFT</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
