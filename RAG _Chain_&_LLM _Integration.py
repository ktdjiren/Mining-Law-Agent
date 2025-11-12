"""
 Step 2: RAG Chain & LLM Integration
============================================================
Author: [Aman Rawat]
Description: Connects embeddings to LLM for intelligent question answering

This script:
1. Loads the embeddings/vectorstore 
2. Sets up retrieval chain
3. Integrates with LLM (multiple options)
4. Creates question-answering pipeline
"""

import os
import sys
import json
import warnings
from typing import List, Dict, Optional
from datetime import datetime
import argparse

warnings.filterwarnings('ignore')

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

# LLM imports (multiple options)
try:
    from langchain_community.llms import HuggingFaceHub
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RAGConfig:
    """Configuration for RAG system"""
    
    def __init__(self, 
                 vectorstore_path: str = "./data/processed/mining_laws_vectorstore",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_provider: str = "huggingface"):
        
        self.VECTORSTORE_PATH = vectorstore_path
        self.EMBEDDING_MODEL = embedding_model
        self.LLM_PROVIDER = llm_provider.lower()
        
        # Retrieval settings
        self.TOP_K_RESULTS = 5
        self.SEARCH_TYPE = "similarity"  # or "mmr"
        
        # LLM settings by provider
        self.LLM_CONFIGS = {
            'huggingface': {
                'model': 'mistralai/Mistral-7B-Instruct-v0.2',
                'temperature': 0.3,
                'max_length': 1024,
                'max_new_tokens': 512
            },
            'groq': {
                'model': 'llama-3.1-70b-versatile',
                'temperature': 0.3,
                'max_tokens': 1024
            },
            'openai': {
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.3,
                'max_tokens': 1024
            }
        }
        
        # Output paths
        self.OUTPUT_DIR = "./outputs"
        self.QUERIES_LOG = os.path.join(self.OUTPUT_DIR, "queries_log.json")
        
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
    def print_config(self):
        """Print configuration"""
        print("\n" + "="*80)
        print("  RAG SYSTEM CONFIGURATION")
        print("="*80)
        print(f" Vector Store:       {self.VECTORSTORE_PATH}")
        print(f" Embedding Model:    {self.EMBEDDING_MODEL}")
        print(f" LLM Provider:       {self.LLM_PROVIDER}")
        print(f" Top-K Results:      {self.TOP_K_RESULTS}")
        print(f" Search Type:        {self.SEARCH_TYPE}")
        
        if self.LLM_PROVIDER in self.LLM_CONFIGS:
            config = self.LLM_CONFIGS[self.LLM_PROVIDER]
            print(f" LLM Model:          {config.get('model', 'N/A')}")
            print(f"  Temperature:        {config.get('temperature', 'N/A')}")
        
        print("="*80 + "\n")


class MiningLawsRAG:
    """Complete RAG system for mining laws"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Load embeddings model
        print(" Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        print(" Embedding model loaded\n")
        
        # Load vector store
        self.vectorstore = self._load_vectorstore()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Create RAG chain
        self.qa_chain = self._create_rag_chain()
        
        # Query history
        self.query_history = []
    
    def _load_vectorstore(self) -> Chroma:
        """Load the vector store"""
        print(f" Loading vector store from: {self.config.VECTORSTORE_PATH}")
        
        if not os.path.exists(self.config.VECTORSTORE_PATH):
            raise FileNotFoundError(
                f"Vector store not found at {self.config.VECTORSTORE_PATH}\n"
                "Please run Step 1 (process_documents.py) first!"
            )
        
        vectorstore = Chroma(
            persist_directory=self.config.VECTORSTORE_PATH,
            embedding_function=self.embeddings,
            collection_name="mining_laws"
        )
        
        print(" Vector store loaded\n")
        return vectorstore
    
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        print(f" Initializing LLM: {self.config.LLM_PROVIDER}")
        
        provider = self.config.LLM_PROVIDER
        config = self.config.LLM_CONFIGS.get(provider, {})
        
        if provider == 'huggingface':
            if not HF_AVAILABLE:
                raise ImportError("HuggingFace not available. Install: pip install huggingface-hub")
            
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not api_key:
                raise ValueError(
                    "HUGGINGFACE_API_KEY not found in environment.\n"
                    "Get your key from: https://huggingface.co/settings/tokens\n"
                    "Set it: export HUGGINGFACE_API_KEY='your_key'"
                )
            
            llm = HuggingFaceHub(
                repo_id=config['model'],
                huggingfacehub_api_token=api_key,
                model_kwargs={
                    'temperature': config['temperature'],
                    'max_length': config['max_length'],
                    'max_new_tokens': config['max_new_tokens']
                }
            )
        
        elif provider == 'groq':
            if not GROQ_AVAILABLE:
                raise ImportError("Groq not available. Install: pip install groq langchain-groq")
            
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not found in environment.\n"
                    "Get your key from: https://console.groq.com/keys\n"
                    "Set it: export GROQ_API_KEY='your_key'"
                )
            
            llm = ChatGroq(
                model=config['model'],
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                groq_api_key=api_key
            )
        
        elif provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install: pip install openai langchain-openai")
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment.\n"
                    "Get your key from: https://platform.openai.com/api-keys\n"
                    "Set it: export OPENAI_API_KEY='your_key'"
                )
            
            llm = ChatOpenAI(
                model=config['model'],
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                openai_api_key=api_key
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        print(f" LLM initialized: {config['model']}\n")
        return llm
    
    def _create_rag_chain(self):
        """Create the RAG chain"""
        print(" Creating RAG chain...")
        
        # Custom prompt template
        template = """You are an expert AI assistant specialized in Indian mining laws and regulations.
                      Your role is to provide accurate, detailed answers based ONLY on the mining legislation documents provided.
                      Context from mining law documents:
                      {context}
                      
                      Question: {question}
                      
                      CRITICAL INSTRUCTIONS:
                      1. Answer ONLY based on the context provided above
                      2. Cite specific acts, sections, regulations, or circulars with exact names
                      3. If the answer is not in the context, clearly state: "I don't have information about this in the available documents. Please ask questions related to Indian mining laws, regulations, safety standards, or compliance requirements."
                      4. Be precise and use proper legal terminology
                      5. Structure your answer clearly with:
                         - Direct answer first
                         - Relevant legal references
                         - Additional context if helpful
                      6. If multiple documents address the question, mention all relevant sources
                      7. For safety or compliance questions, emphasize the specific requirements
                      
                      Detailed Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type=self.config.SEARCH_TYPE,
            search_kwargs={"k": self.config.TOP_K_RESULTS}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print(" RAG chain created\n")
        return qa_chain
    
    def query(self, question: str, show_sources: bool = True, save_log: bool = True) -> Dict:
        """Query the RAG system"""
        
        print("\n" + "="*80)
        print(f"❓ QUESTION: {question}")
        print("="*80 + "\n")
        print("🔍 Searching documents and generating answer...\n")
        
        try:
            # Get answer from RAG chain
            result = self.qa_chain({"query": question})
            
            answer = result['result']
            source_docs = result['source_documents']
            
            # Format answer
            print(" ANSWER:")
            print("-" * 80)
            print(answer)
            print("-" * 80 + "\n")
            
            # Show sources
            if show_sources:
                self._display_sources(source_docs)
            
            # Create result object
            query_result = {
                'question': question,
                'answer': answer,
                'sources': [
                    {
                        'source': doc.metadata['source'],
                        'doc_type': doc.metadata['doc_type'],
                        'subject': doc.metadata['subject'],
                        'year': doc.metadata.get('year', 'Unknown'),
                        'chunk_id': doc.metadata['chunk_id']
                    }
                    for doc in source_docs
                ],
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            # Save to history
            self.query_history.append(query_result)
            
            # Save log
            if save_log:
                self._save_query_log(query_result)
            
            return query_result
            
        except Exception as e:
            error_msg = f"Error occurred: {str(e)}"
            print(f"\n {error_msg}\n")
            
            query_result = {
                'question': question,
                'answer': error_msg,
                'sources': [],
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            
            return query_result
    
    def _display_sources(self, source_docs: List[Document]):
        """Display source documents"""
        print(" SOURCES:")
        print("="*80)
        
        seen_sources = set()
        for idx, doc in enumerate(source_docs, 1):
            source_key = f"{doc.metadata['source']}-{doc.metadata['chunk_id']}"
            
            if source_key not in seen_sources:
                print(f"\n[{idx}] {doc.metadata['source']}")
                print(f"    Type: {doc.metadata['doc_type']}")
                print(f"    Subject: {doc.metadata['subject']}")
                print(f"    Year: {doc.metadata.get('year', 'Unknown')}")
                print(f"    Chunk: {doc.metadata['chunk_id']}/{doc.metadata['chunk_total']}")
                print(f"    Preview: {doc.page_content[:150]}...")
                seen_sources.add(source_key)
        
        print("\n" + "="*80 + "\n")
    
    def _save_query_log(self, query_result: Dict):
        """Save query to log file"""
        # Load existing logs
        if os.path.exists(self.config.QUERIES_LOG):
            with open(self.config.QUERIES_LOG, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new query
        logs.append(query_result)
        
        # Save updated logs
        with open(self.config.QUERIES_LOG, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def batch_query(self, questions: List[str]):
        """Run multiple queries"""
        print(f"\n Running batch queries ({len(questions)} questions)...\n")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"Query {i}/{len(questions)}")
            print(f"{'='*80}")
            
            result = self.query(question, show_sources=False)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get query statistics"""
        if not self.query_history:
            return {"message": "No queries yet"}
        
        successful = sum(1 for q in self.query_history if q['success'])
        failed = len(self.query_history) - successful
        
        return {
            'total_queries': len(self.query_history),
            'successful': successful,
            'failed': failed,
            'success_rate': f"{(successful/len(self.query_history)*100):.1f}%"
        }


def run_interactive_mode(rag_system: MiningLawsRAG):
    """Interactive question-answering mode"""
    print("\n" + "="*80)
    print(" INTERACTIVE MODE")
    print("="*80)
    print("\nAsk questions about Indian mining laws and regulations.")
    print("Type 'exit', 'quit', or 'q' to exit.")
    print("Type 'stats' to see query statistics.\n")
    
    while True:
        try:
            question = input(" Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n Goodbye!\n")
                break
            
            if question.lower() == 'stats':
                stats = rag_system.get_statistics()
                print("\n Query Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                print()
                continue
            
            rag_system.query(question)
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!\n")
            break
        except Exception as e:
            print(f"\n Error: {e}\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Mining Laws RAG System - Question Answering'
    )
    parser.add_argument(
        '--vectorstore-path',
        type=str,
        default='./data/processed/mining_laws_vectorstore',
        help='Path to vector store'
    )
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['huggingface', 'groq', 'openai'],
        default='huggingface',
        help='LLM provider to use'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--questions',
        type=str,
        nargs='+',
        help='Questions to answer'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*80)
    print("⛏️  MINING LAWS RAG SYSTEM - QUESTION ANSWERING")
    print("="*80)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize configuration
    config = RAGConfig(
        vectorstore_path=args.vectorstore_path,
        llm_provider=args.llm_provider
    )
    config.print_config()
    
    # Initialize RAG system
    try:
        rag_system = MiningLawsRAG(config)
    except Exception as e:
        print(f" Failed to initialize RAG system: {e}")
        return
    
    # Run based on mode
    if args.interactive:
        run_interactive_mode(rag_system)
    
    elif args.questions:
        # Answer provided questions
        for question in args.questions:
            rag_system.query(question)
    
    else:
        # Default: run sample questions
        print(" Running sample questions...\n")
        
        sample_questions = [
            "What are the main provisions of the Mines Act 1952?",
            "What is the minimum age for employment in mines?",
            "What are the safety requirements for underground coal mines?",
            "What are the penalties for violating mining regulations?",
            "What is DGMS and what are its functions?"
        ]
        
        for question in sample_questions:
            rag_system.query(question)
            print("\n" + "="*80 + "\n")
        
        # Show statistics
        stats = rag_system.get_statistics()
        print("\n SESSION STATISTICS")
        print("="*80)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("="*80 + "\n")
    
    print(f" Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Query logs saved to: {config.QUERIES_LOG}\n")


if __name__ == "__main__":
    main()
