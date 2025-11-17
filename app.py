"""
Pseudo-code to Code Generator - Streamlit App
Fine-tuned GPT-2 with LoRA on SPoC Dataset
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Pseudo-code to Code Generator",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Text areas */
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    
    h2 {
        color: #ff7f0e;
        font-weight: 600;
    }
    
    h3 {
        color: #2ca02c;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_model(model_name="manesh230/pseudo-code-generator"):
    """
    Load the fine-tuned GPT-2 model with LoRA weights from Hugging Face Hub
    
    Args:
        model_name: HuggingFace model repository name
        
    Returns:
        model: Loaded PEFT model
        tokenizer: Loaded tokenizer
        device: Device (cuda/cpu)
    """
    try:
        with st.spinner(f"🔄 Loading model from Hugging Face ({model_name})..."):
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad token (GPT-2 doesn't have one by default)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load base GPT-2 model
            base_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                low_cpu_mem_usage=True
            )
            
            # Load LoRA weights
            model = PeftModel.from_pretrained(
                base_model, 
                model_name,
                torch_dtype=torch.float32
            )
            
            # Move to device
            model = model.to(device)
            model.eval()  # Set to evaluation mode
            
            return model, tokenizer, device
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info(f"""
        **Troubleshooting:**
        1. Make sure the model exists at: https://huggingface.co/{model_name}
        2. Check that the model is public
        3. Verify you uploaded the model correctly
        """)
        st.stop()

# ============================================================================
# CODE GENERATION FUNCTION
# ============================================================================
def generate_code(model, tokenizer, device, pseudo_code, temperature=0.7, 
                 max_length=150, num_samples=1):
    """
    Generate Python code from pseudo-code description
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        device: Device (cuda/cpu)
        pseudo_code: Input pseudo-code description
        temperature: Sampling temperature (higher = more random)
        max_length: Maximum tokens to generate
        num_samples: Number of different outputs
        
    Returns:
        List of generated code snippets
    """
    try:
        # Format the prompt
        prompt = f"Pseudo: {pseudo_code} Code:"
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)
        
        # Generate code
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Reduce repetitive outputs
                no_repeat_ngram_size=3   # Avoid repeating 3-grams
            )
        
        # Decode outputs
        generated_codes = []
        for output in outputs:
            # Decode the tokens
            text = tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract only the code part (after "Code:")
            if "Code:" in text:
                code = text.split("Code:")[1].strip()
            else:
                code = text.strip()
            
            # Clean up any remaining pseudo-code artifacts
            code = code.replace("Pseudo:", "").strip()
            
            # Only add non-empty results
            if code:
                generated_codes.append(code)
        
        return generated_codes if generated_codes else ["# Error: Could not generate code"]
    
    except Exception as e:
        st.error(f"❌ Error during generation: {str(e)}")
        return [f"# Error: {str(e)}"]

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # ========================================================================
    # HEADER
    # ========================================================================
    st.title("💻 Pseudo-code to Code Generator")
    st.markdown("""
    ### 🚀 Transform natural language descriptions into Python code using AI
    
    This application uses a **GPT-2 model fine-tuned with LoRA** on the 
    [SPoC dataset](https://github.com/sumith1896/spoc) to generate executable 
    Python code from pseudo-code descriptions.
    """)
    
    st.markdown("---")
    
    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    # Show initial loading message
    if 'model_loaded' not in st.session_state:
        st.info("🔄 Loading AI model... This may take 1-2 minutes on first run.")
    
    # Load the model (cached after first load)
    model, tokenizer, device = load_model("manesh230/pseudo-code-generator")
    
    # Mark as loaded
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = True
        st.success("✅ Model loaded successfully!")
        time.sleep(1)
        st.rerun()
    
    # ========================================================================
    # SIDEBAR - CONFIGURATION
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        st.markdown("#### 🎛️ Model Parameters")
        
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.1,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Controls randomness:\n- Lower (0.1-0.5): More focused and deterministic\n- Higher (0.8-1.5): More creative and diverse"
        )
        
        max_length = st.slider(
            "📏 Max Length",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Maximum number of tokens (words) to generate"
        )
        
        num_samples = st.slider(
            "🔢 Number of Outputs",
            min_value=1,
            max_value=3,
            value=1,
            help="Generate multiple different code variations"
        )
        
        st.markdown("---")
        
        # Model Information
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Architecture:** GPT-2 + LoRA  
        **Base Model:** GPT-2 (124M params)  
        **Fine-tuning:** LoRA (294K trainable)  
        **Training Data:** 20,000 samples  
        **Eval Loss:** 1.16  
        **BLEU Score:** 7.89  
        **Device:** {device.upper()}  
        """)
        
        st.markdown("---")
        
        # About Section
        st.markdown("### 📚 About This Project")
        st.markdown("""
        **Fine-tuning Details:**
        - **Method:** LoRA (Low-Rank Adaptation)
        - **Dataset:** SPoC (Pseudocode to Code)
        - **Epochs:** 2
        - **Batch Size:** 16 (effective)
        - **Learning Rate:** 5e-4
        
        **Performance Metrics:**
        - BLEU-1: 10.23
        - BLEU-2: 8.51
        - BLEU-3: 7.24
        - BLEU-4: 6.14
        - Partial Match: 81%
        """)
        
        st.markdown("---")
        
        # Links
        st.markdown("### 🔗 Links")
        st.markdown("""
        - [📄 Research Paper](https://arxiv.org/pdf/1906.04908)
        - [💾 Dataset](https://github.com/sumith1896/spoc)
        - [🤗 Model](https://huggingface.co/manesh230/pseudo-code-generator)
        """)
    
    # ========================================================================
    # MAIN CONTENT - TWO COLUMNS
    # ========================================================================
    col1, col2 = st.columns([1, 1], gap="large")
    
    # ========================================================================
    # LEFT COLUMN - INPUT
    # ========================================================================
    with col1:
        st.subheader("📝 Input Pseudo-code")
        
        # Example selector
        st.markdown("**💡 Quick Examples:**")
        
        examples = {
            "➡️ Custom input": "",
            "➕ Add two numbers": "create a function that adds two numbers",
            "🔁 Print loop": "write a loop that prints numbers from 1 to 10",
            "👤 Student class": "define a class for a student with name and age",
            "🔄 Reverse string": "create a function to reverse a string",
            "🔍 Binary search": "implement binary search algorithm",
            "✖️ Factorial": "write a recursive function to calculate factorial",
            "📊 Sort list": "create a function to sort a list in ascending order",
            "🔝 Find maximum": "write a function to find the maximum value in a list",
            "🔢 Even numbers": "create a function that returns all even numbers from a list",
            "📝 Count words": "write a function to count words in a string",
            "🔐 Check palindrome": "implement a function to check if a string is a palindrome"
        }
        
        example_choice = st.selectbox(
            "Select an example or write your own below:",
            list(examples.keys()),
            label_visibility="visible"
        )
        
        default_value = examples[example_choice]
        
        # Text input area
        pseudo_code = st.text_area(
            "Enter your pseudo-code description:",
            value=default_value,
            height=200,
            placeholder="Example: create a function that calculates the sum of all even numbers in a list",
            help="Describe what you want the code to do in plain English"
        )
        
        # Character count
        char_count = len(pseudo_code)
        st.caption(f"Characters: {char_count}")
        
        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
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
        
        with col_btn3:
            if st.button("ℹ️ Help", use_container_width=True):
                st.info("""
                **Tips for better results:**
                - Be specific and clear
                - Use simple language
                - Mention input/output types
                - Keep it concise
                
                **Example:**
                ✅ "create a function that takes a list of numbers and returns the sum of even numbers"
                ❌ "do something with numbers"
                """)
        
        if clear_button:
            st.rerun()
    
    # ========================================================================
    # RIGHT COLUMN - OUTPUT
    # ========================================================================
    with col2:
        st.subheader("💻 Generated Code")
        
        if generate_button:
            if not pseudo_code.strip():
                st.warning("⚠️ Please enter a pseudo-code description!")
            else:
                # Generation process
                with st.spinner(f"🔄 Generating {num_samples} code sample(s)..."):
                    start_time = time.time()
                    
                    # Generate code
                    codes = generate_code(
                        model,
                        tokenizer,
                        device,
                        pseudo_code,
                        temperature,
                        max_length,
                        num_samples
                    )
                    
                    end_time = time.time()
                    generation_time = end_time - start_time
                
                # Display results
                if codes and codes[0] != "# Error: Could not generate code":
                    st.success(f"✅ Generated in {generation_time:.2f} seconds!")
                    
                    for i, code in enumerate(codes, 1):
                        # Sample header
                        if num_samples > 1:
                            st.markdown(f"#### 📄 Sample {i} of {num_samples}")
                        
                        # Display code with syntax highlighting
                        st.code(code, language="python", line_numbers=True)
                        
                        # Statistics
                        code_lines = code.count('\n') + 1
                        code_chars = len(code)
                        code_words = len(code.split())
                        
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        with col_stat1:
                            st.metric("Lines", code_lines)
                        with col_stat2:
                            st.metric("Words", code_words)
                        with col_stat3:
                            st.metric("Chars", code_chars)
                        with col_stat4:
                            st.download_button(
                                label="📥 Download",
                                data=code,
                                file_name=f"generated_code_{i}.py",
                                mime="text/plain",
                                key=f"download_{i}",
                                use_container_width=True
                            )
                        
                        # Separator between samples
                        if i < len(codes):
                            st.markdown("---")
                    
                    # Overall statistics
                    avg_length = sum(len(c.split()) for c in codes) / len(codes)
                    st.info(f"📊 Generated {len(codes)} sample(s) • Avg: {avg_length:.0f} tokens • Time: {generation_time:.2f}s")
                else:
                    st.error("❌ Failed to generate code. Please try again with different settings or input.")
        else:
            # Initial state - show instructions
            st.info("""
            👈 **How to use:**
            
            1. Select an example or write your own pseudo-code
            2. Adjust generation settings in the sidebar (optional)
            3. Click **"Generate Code"** button
            4. View and download the generated Python code
            
            **First time?** Try the examples to see what the model can do!
            """)
            
            # Show example output
            with st.expander("👁️ See Example Output"):
                st.markdown("**Input:** `create a function that adds two numbers`")
                st.code("""def add_numbers(a, b):
    return a + b""", language="python")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        <p><strong>Pseudo-code to Code Generator</strong></p>
        <p>Built with ❤️ using <strong>GPT-2 + LoRA</strong> | Fine-tuned on <strong>SPoC Dataset</strong></p>
        <p>
            <a href="https://streamlit.io" target="_blank">Streamlit</a> • 
            <a href="https://huggingface.co/transformers" target="_blank">Transformers</a> • 
            <a href="https://pytorch.org" target="_blank">PyTorch</a> • 
            <a href="https://github.com/huggingface/peft" target="_blank">PEFT</a>
        </p>
        <p style="font-size: 12px; color: #999;">
            Model performance: BLEU 7.89 | Partial Match 81% | 294K trainable parameters
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()
