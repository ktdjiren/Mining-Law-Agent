
Dataset: Indian Mining Laws & Regulations (40 documents)
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import argparse

# PDF Processing
import PyPDF2
import fitz  # PyMuPDF

# ML & Embeddings
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

warnings.filterwarnings('ignore')


class Config:
    """Configuration for Mining Laws RAG System"""
    
    def __init__(self, 
                 data_path: str = "./data/raw",
                 output_path: str = "./data/processed",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        # Paths
        self.DATA_PATH = data_path
        self.OUTPUT_PATH = output_path
        self.VECTORSTORE_PATH = os.path.join(output_path, "vectorstore")
        self.PROCESSED_TEXT_PATH = os.path.join(output_path, "processed_texts")
        self.METADATA_PATH = os.path.join(output_path, "metadata.json")
        self.EMBEDDING_INFO_PATH = os.path.join(output_path, "embedding_info.json")
        
        # Embedding configuration
        self.EMBEDDING_MODEL = embedding_model
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.EMBEDDING_BATCH_SIZE = 32 if torch.cuda.is_available() else 8
        
        # Text chunking parameters
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        
        # Processing options
        self.USE_PYMUPDF = True
        self.SAVE_PROCESSED_TEXT = True
        
        # Create directories
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        os.makedirs(self.VECTORSTORE_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_TEXT_PATH, exist_ok=True)
    
    def print_config(self):
        """Print configuration details"""
        print("\n" + "="*80)
        print(" CONFIGURATION")
        print("="*80)
        print(f" Data Path:          {self.DATA_PATH}")
        print(f" Output Path:        {self.OUTPUT_PATH}")
        print(f" Vector Store:       {self.VECTORSTORE_PATH}")
        print(f" Embedding Model:    {self.EMBEDDING_MODEL}")
        print(f"  Device:             {self.DEVICE}")
        
        if self.DEVICE == 'cuda':
            print(f" GPU:                {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f" GPU Memory:         {gpu_memory:.2f} GB")
            print(f" Batch Size:         {self.EMBEDDING_BATCH_SIZE}")
        
        print(f" Chunk Size:         {self.CHUNK_SIZE}")
        print(f" Chunk Overlap:      {self.CHUNK_OVERLAP}")
        print("="*80 + "\n")


class PDFTextExtractor:
    """Extract and clean text from PDF documents"""
    
    def __init__(self, use_pymupdf: bool = True):
        self.use_pymupdf = use_pymupdf
    
    def extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (better quality)"""
        try:
            text = ""
            doc = fitz.open(pdf_path)
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"  PyMuPDF failed for {pdf_path}: {e}")
            return ""
    
    def extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback)"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    try:
                        text += page.extract_text() + "\n"
                    except:
                        continue
            
            return text.strip()
        except Exception as e:
            print(f"  PyPDF2 failed for {pdf_path}: {e}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text with fallback methods"""
        if self.use_pymupdf:
            text = self.extract_with_pymupdf(pdf_path)
            if text and len(text) > 100:
                return text
        
        text = self.extract_with_pypdf2(pdf_path)
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters, keep printable and newlines
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text


class DocumentClassifier:
    """Classify mining law documents by type, subject, and year"""
    
    @staticmethod
    def classify_document(filename: str) -> Dict[str, str]:
        """Classify document and extract metadata from filename"""
        import re
        filename_lower = filename.lower()
        
        # Document type classification
        if 'act' in filename_lower and 'report' not in filename_lower:
            doc_type = 'Act'
        elif 'regulation' in filename_lower or 'rule' in filename_lower:
            doc_type = 'Regulation'
        elif 'circular' in filename_lower:
            doc_type = 'Circular'
        elif 'amendment' in filename_lower:
            doc_type = 'Amendment'
        elif 'report' in filename_lower:
            doc_type = 'Report'
        elif 'policy' in filename_lower:
            doc_type = 'Policy'
        elif 'guideline' in filename_lower:
            doc_type = 'Guideline'
        elif 'scheme' in filename_lower:
            doc_type = 'Scheme'
        elif 'development' in filename_lower:
            doc_type = 'Development Plan'
        else:
            doc_type = 'Document'
        
        # Extract year
        year_match = re.search(r'(19|20)\d{2}', filename)
        year = year_match.group() if year_match else 'Unknown'
        
        # Subject classification
        subject = DocumentClassifier._classify_subject(filename_lower)
        
        return {
            'doc_type': doc_type,
            'year': year,
            'subject': subject
        }
    
    @staticmethod
    def _classify_subject(filename: str) -> str:
        """Classify document subject area"""
        subject_keywords = {
            'Coal Mining': ['coal'],
            'Atomic Minerals': ['atomic', 'energy'],
            'Safety & Inspection': ['dgms', 'safety', 'rescue'],
            'Labour & Employment': ['employment', 'women', 'worker'],
            'Welfare & Benefits': ['pension', 'provident', 'welfare', 'creche'],
            'Electricity & Energy': ['electricity', 'power'],
            'Mineral Development': ['mineral', 'development'],
            'Training & Education': ['vocational', 'training', 'education'],
            'Legal & Judicial': ['court', 'enquiry', 'appeal'],
            'Environmental': ['environment', 'forest', 'pollution']
        }
        
        for subject, keywords in subject_keywords.items():
            if any(kw in filename for kw in keywords):
                return subject
        
        return 'General Mining'


class MiningLawsProcessor:
    """Complete pipeline for processing mining law documents"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pdf_extractor = PDFTextExtractor(use_pymupdf=config.USE_PYMUPDF)
        
        # Initialize embeddings model
        print(f"\n Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.DEVICE},
            encode_kwargs={
                'batch_size': config.EMBEDDING_BATCH_SIZE,
                'normalize_embeddings': True,
                'show_progress_bar': True
            }
        )
        print(" Embedding model loaded\n")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        # Statistics
        self.document_metadata = {}
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'total_chars': 0,
            'total_words': 0
        }
    
    def process_single_document(self, pdf_path: str) -> Tuple[List[Document], Dict]:
        """Process a single PDF document"""
        filename = os.path.basename(pdf_path)
        
        # Extract text
        text = self.pdf_extractor.extract_text(pdf_path)
        
        if not text or len(text) < 100:
            return [], {'error': 'Insufficient text extracted'}
        
        # Clean text
        text = self.pdf_extractor.clean_text(text)
        
        # Classify document
        classification = DocumentClassifier.classify_document(filename)
        
        # Create metadata
        base_metadata = {
            'source': filename,
            'path': pdf_path,
            'doc_type': classification['doc_type'],
            'year': classification['year'],
            'subject': classification['subject'],
            'char_count': len(text),
            'word_count': len(text.split()),
            'processed_date': datetime.now().isoformat()
        }
        
        # Save processed text
        if self.config.SAVE_PROCESSED_TEXT:
            text_file = os.path.join(
                self.config.PROCESSED_TEXT_PATH,
                filename.replace('.pdf', '.txt')
            )
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects
        documents = []
        for idx, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': idx,
                'chunk_total': len(chunks),
                'chunk_chars': len(chunk)
            })
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        # Update statistics
        self.stats['total_chunks'] += len(chunks)
        self.stats['total_chars'] += len(text)
        self.stats['total_words'] += len(text.split())
        
        return documents, base_metadata
    
    def process_all_documents(self) -> List[Document]:
        """Process all PDF documents in the dataset"""
        print("\n" + "="*80)
        print(" STARTING DOCUMENT PROCESSING")
        print("="*80 + "\n")
        
        all_documents = []
        
        # Find all PDF files
        pdf_files = []
        for root, dirs, files in os.walk(self.config.DATA_PATH):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        self.stats['total_files'] = len(pdf_files)
        
        if not pdf_files:
            print(f" No PDF files found in {self.config.DATA_PATH}")
            return []
        
        print(f" Found {len(pdf_files)} PDF files")
        print(f" Processing from: {self.config.DATA_PATH}\n")
        
        # Process each file
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            filename = os.path.basename(pdf_path)
            
            try:
                documents, metadata = self.process_single_document(pdf_path)
                
                if documents:
                    all_documents.extend(documents)
                    self.document_metadata[filename] = metadata
                    self.stats['successful'] += 1
                    tqdm.write(f" {filename}: {len(documents)} chunks")
                else:
                    self.stats['failed'] += 1
                    tqdm.write(f" {filename}: Failed")
                    
            except Exception as e:
                self.stats['failed'] += 1
                tqdm.write(f" {filename}: {str(e)}")
        
        self._print_summary()
        self._save_metadata()
        
        return all_documents
    
    def _print_summary(self):
        """Print processing statistics"""
        print("\n" + "="*80)
        print(" PROCESSING SUMMARY")
        print("="*80)
        print(f"Total Files:           {self.stats['total_files']}")
        print(f" Successful:          {self.stats['successful']}")
        print(f" Failed:              {self.stats['failed']}")
        print(f" Total Chunks:        {self.stats['total_chunks']:,}")
        print(f" Total Characters:    {self.stats['total_chars']:,}")
        print(f" Total Words:         {self.stats['total_words']:,}")
        if self.stats['successful'] > 0:
            avg_chunks = self.stats['total_chunks'] / self.stats['successful']
            print(f" Avg Chunks/Doc:      {avg_chunks:.1f}")
        print("="*80 + "\n")
    
    def _save_metadata(self):
        """Save processing metadata"""
        metadata = {
            'processing_stats': self.stats,
            'documents': self.document_metadata,
            'config': {
                'embedding_model': self.config.EMBEDDING_MODEL,
                'chunk_size': self.config.CHUNK_SIZE,
                'chunk_overlap': self.config.CHUNK_OVERLAP,
                'device': self.config.DEVICE
            },
            'processed_date': datetime.now().isoformat()
        }
        
        with open(self.config.METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f" Metadata saved: {self.config.METADATA_PATH}")
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create embeddings and vector store"""
        print("\n" + "="*80)
        print(" CREATING EMBEDDINGS & VECTOR STORE")
        print("="*80)
        print(f" Document chunks: {len(documents)}")
        print(f" Embedding model: {self.config.EMBEDDING_MODEL}")
        print(f" Output path: {self.config.VECTORSTORE_PATH}")
        print("\n This may take several minutes...\n")
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.config.VECTORSTORE_PATH,
            collection_name="mining_laws"
        )
        
        print("\n Vector store created successfully!")
        print(f" Total embeddings: {len(documents)}")
        print("="*80 + "\n")
        
        # Save embedding info
        self._save_embedding_info(documents)
        
        return vectorstore
    
    def _save_embedding_info(self, documents: List[Document]):
        """Save embedding information"""
        embedding_info = {
            'vectorstore_path': self.config.VECTORSTORE_PATH,
            'embedding_model': self.config.EMBEDDING_MODEL,
            'total_documents': len(documents),
            'chunk_size': self.config.CHUNK_SIZE,
            'chunk_overlap': self.config.CHUNK_OVERLAP,
            'device': self.config.DEVICE,
            'created_date': datetime.now().isoformat(),
            'ready_for_llm': True
        }
        
        with open(self.config.EMBEDDING_INFO_PATH, 'w') as f:
            json.dump(embedding_info, f, indent=2)
        
        print(f" Embedding info saved: {self.config.EMBEDDING_INFO_PATH}")
    
    def load_vectorstore(self) -> Chroma:
        """Load existing vector store"""
        print(f"\n Loading vector store from: {self.config.VECTORSTORE_PATH}")
        
        if not os.path.exists(self.config.VECTORSTORE_PATH):
            raise FileNotFoundError(
                f"Vector store not found. Run processing first."
            )
        
        vectorstore = Chroma(
            persist_directory=self.config.VECTORSTORE_PATH,
            embedding_function=self.embeddings,
            collection_name="mining_laws"
        )
        
        print(" Vector store loaded\n")
        return vectorstore


def test_similarity_search(vectorstore: Chroma, query: str, k: int = 3):
    """Test similarity search"""
    print(f"\n Query: '{query}'")
    print(f" Top {k} results:\n")
    
    results = vectorstore.similarity_search(query, k=k)
    
    for idx, doc in enumerate(results, 1):
        print(f"[{idx}] {doc.metadata['source']}")
        print(f"    Type: {doc.metadata['doc_type']} | Subject: {doc.metadata['subject']}")
        print(f"    Year: {doc.metadata['year']}")
        print(f"    Preview: {doc.page_content[:150]}...\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Mining Laws RAG System - Document Processing & Embedding Creation'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data/raw',
        help='Path to raw PDF documents'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./data/processed',
        help='Path for processed outputs'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Hugging Face embedding model name'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run similarity search tests after processing'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*80)
    print("  MINING LAWS RAG SYSTEM - DOCUMENT PROCESSOR")
    print("="*80)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize configuration
    config = Config(
        data_path=args.data_path,
        output_path=args.output_path,
        embedding_model=args.embedding_model
    )
    config.print_config()
    
    # Initialize processor
    processor = MiningLawsProcessor(config)
    
    # Process documents
    documents = processor.process_all_documents()
    
    if not documents:
        print(" No documents processed. Exiting.")
        return
    
    # Create embeddings and vector store
    vectorstore = processor.create_vectorstore(documents)
    
    # Run tests if requested
    if args.test:
        print("\n" + "="*80)
        print(" RUNNING SIMILARITY SEARCH TESTS")
        print("="*80)
        
        test_queries = [
            "What are the safety requirements for coal mines?",
            "minimum age for workers in mines",
            "DGMS regulations and compliance"
        ]
        
        for query in test_queries:
            test_similarity_search(vectorstore, query)
            print("-" * 80)
    
    # Final summary
    print("\n" + "="*80)
    print(" PROCESSING COMPLETE!")
    print("="*80)
    print(f" {len(documents)} document chunks embedded")
    print(f" Vector store: {config.VECTORSTORE_PATH}")
    print(f" Processed texts: {config.PROCESSED_TEXT_PATH}")
    print(f" Metadata: {config.METADATA_PATH}")
    print(f" Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(" GPU cache cleared\n")


if __name__ == "__main__":
    main()
