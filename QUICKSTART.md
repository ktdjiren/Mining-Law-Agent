# 🚀 Quick Start Guide

Get your Mining Laws RAG system up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Git installed
- 8GB RAM (16GB recommended)
- GPU optional (3-4x faster)

## Step 1: Clone & Setup (2 minutes)

```bash
# Clone the repository
git clone https://github.com/[your-username]/mining-laws-rag.git
cd mining-laws-rag

# Run setup script (Linux/Mac)
chmod +x setup.sh
./setup.sh

# Or manual setup (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Add Your Data (1 minute)

```bash
# Copy your PDF files to the data directory
cp /path/to/your/pdfs/*.pdf data/raw/

# Or manually:
# - Create data/raw/ folder
# - Copy all 40 PDF files there
```

## Step 3: Process Documents (5-10 minutes)

```bash
# Activate virtual environment (if not already)
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Run processing
python process_documents.py --test
```

**What happens:**
- ✅ Extracts text from all PDFs
- ✅ Creates 1000-character chunks
- ✅ Generates embeddings using sentence-transformers
- ✅ Stores in ChromaDB vector database
- ✅ Runs test queries

## Step 4: Verify Setup

After processing completes, you should see:

```
🎉 PROCESSING COMPLETE!
================================================================================
✅ 1,247 document chunks embedded
💾 Vector store: ./data/processed/vectorstore
📄 Processed texts: ./data/processed/processed_texts
📊 Metadata: ./data/processed/metadata.json
================================================================================
```

Check the outputs:
```bash
ls -lh data/processed/
# Should show:
# - vectorstore/
# - processed_texts/
# - metadata.json
# - embedding_info.json
```

## Step 5: Test Queries

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector store
vectorstore = Chroma(
    persist_directory="./data/processed/vectorstore",
    embedding_function=embeddings,
    collection_name="mining_laws"
)

# Query
query = "What are the safety requirements for coal mines?"
results = vectorstore.similarity_search(query, k=3)

# Display results
for i, doc in enumerate(results, 1):
    print(f"\n[{i}] {doc.metadata['source']}")
    print(f"Type: {doc.metadata['doc_type']}")
    print(f"Preview: {doc.page_content[:200]}...")
```

## Common Issues & Solutions

### Issue: "No PDFs found"
**Solution:** Make sure PDFs are in `data/raw/` directory
```bash
ls data/raw/*.pdf  # Should list your PDFs
```

### Issue: "CUDA out of memory"
**Solution:** Use CPU instead or reduce batch size
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU
python process_documents.py
```

### Issue: "Slow processing"
**Solution:** Enable GPU in your environment
```python
import torch
print(torch.cuda.is_available())  # Should return True
```

### Issue: "Import errors"
**Solution:** Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

1. **Integrate with LLM** - Connect to GPT-4, Claude, or open-source models
2. **Build UI** - Create a Gradio or Streamlit interface
3. **Add RAG Chain** - Implement the full question-answering pipeline
4. **Fine-tune** - Optimize chunk sizes and retrieval parameters
5. **Deploy** - Host on Hugging Face Spaces or Streamlit Cloud

## Performance Benchmarks

### Your Setup (40 PDFs):

| Component | Time (GPU) | Time (CPU) |
|-----------|-----------|-----------|
| PDF Extraction | 3-5 min | 3-5 min |
| Chunking | 30 sec | 30 sec |
| Embeddings | 2-3 min | 10-15 min |
| **Total** | **6-9 min** | **14-21 min** |

### Resource Usage:

| Resource | Requirement |
|----------|-------------|
| RAM | 4-8 GB |
| GPU VRAM | 2-4 GB (optional) |
| Disk Space | 500 MB |
| Network | Download models (~400 MB) |

## Command Reference

```bash
# Basic processing
python process_documents.py

# With custom paths
python process_documents.py \
    --data-path /path/to/pdfs \
    --output-path /path/to/output

# With different embedding model
python process_documents.py \
    --embedding-model sentence-transformers/all-mpnet-base-v2

# With testing
python process_documents.py --test

# Show help
python process_documents.py --help
```

## Project Structure

```
mining-laws-rag/
├── process_documents.py       # ← Main script to run
├── requirements.txt
├── README.md
├── QUICKSTART.md             # ← You are here
├── data/
│   ├── raw/                  # ← Put PDFs here
│   └── processed/
│       ├── vectorstore/      # ← Embeddings stored here
│       ├── processed_texts/
│       └── metadata.json
└── venv/                     # Virtual environment
```

## Getting Help

- **Documentation**: See [README.md](README.md)
- **Issues**: Open an issue on GitHub
- **Email**: [your.email@example.com]

## What's Next?

Check out these guides:
1. [RAG_INTEGRATION.md](docs/RAG_INTEGRATION.md) - Connect to LLMs
2. [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deploy your system
3. [API_REFERENCE.md](docs/API_REFERENCE.md) - Code documentation

---

**Need help?** Open an issue or contact the maintainer!

**Happy coding!** ⛏️🚀
