#!/bin/bash

# Mining Laws RAG System - Setup Script
# Run this script to set up your development environment

set -e

echo "⛏️  Mining Laws RAG System - Setup"
echo "=================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.8+ required. You have $python_version"
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed/vectorstore
mkdir -p data/processed/processed_texts
mkdir -p notebooks
mkdir -p logs

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep

echo "✅ Directories created"

# Check GPU availability
echo ""
echo "🎮 Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU detected. Processing will use CPU.")
EOF

# Create sample .env file
echo ""
echo "📝 Creating sample .env file..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOL'
# Mining Laws RAG System - Environment Variables

# Paths
DATA_PATH=./data/raw
OUTPUT_PATH=./data/processed

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API Keys (optional - for LLM integration)
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
# HUGGINGFACE_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here

# Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EOL
    echo "✅ .env file created"
else
    echo "⚠️  .env file already exists. Skipping..."
fi

# Final message
echo ""
echo "=================================="
echo "✅ Setup complete!"
echo "=================================="
echo ""
echo "📋 Next steps:"
echo "1. Copy your PDF files to: data/raw/"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run processing: python process_documents.py"
echo ""
echo "💡 For help: python process_documents.py --help"
echo ""
