# 💻 Pseudo-code to Code Generator

Transform natural language descriptions into Python code using AI!

## 🌟 Features

- **GPT-2 + LoRA**: Fine-tuned model with 294K trainable parameters
- **Real-time Generation**: Generate code instantly from descriptions
- **Multiple Samples**: Get different variations of the same code
- **User-friendly Interface**: Clean and intuitive Streamlit UI

## 🚀 Live Demo

**[Try it on Streamlit Cloud](YOUR_STREAMLIT_URL_HERE)**

## 📊 Model Performance

- **BLEU Score**: 7.89
- **Training Samples**: 20,000
- **Eval Loss**: 1.16
- **Partial Match Rate**: 81%

## 🛠️ Technology Stack

- **Framework**: Streamlit
- **Model**: GPT-2 (124M parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Dataset**: SPoC (Search-based Pseudocode to Code)
- **Libraries**: PyTorch, Transformers, PEFT

## 📦 Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pseudo-code-to-code.git
cd pseudo-code-to-code

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
