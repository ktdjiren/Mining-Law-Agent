
Author: [Aman Rawat]
Description: Interactive web-based chat interface for mining laws Q&A

This script creates a user-friendly chat interface where users can:
1. Ask questions about mining laws
2. View source citations
3. See conversation history
4. Export chat transcripts
"""

import os
import json
import gradio as gr
from datetime import datetime
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Import RAG system from Step 2
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# LLM imports
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


class MiningLawsChatbot:
    """Chatbot interface for Mining Laws RAG system"""
    
    def __init__(self, 
                 vectorstore_path: str = "./data/processed/mining_laws_vectorstore",
                 llm_provider: str = "huggingface"):
        
        self.vectorstore_path = vectorstore_path
        self.llm_provider = llm_provider
        
        # Initialize components
        print(" Initializing chatbot...")
        self.embeddings = self._load_embeddings()
        self.vectorstore = self._load_vectorstore()
        self.llm = self._initialize_llm()
        self.qa_chain = self._create_qa_chain()
        
        # Chat history
        self.conversations = []
        
        print(" Chatbot initialized!\n")
    
    def _load_embeddings(self):
        """Load embedding model"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    def _load_vectorstore(self):
        """Load vector store"""
        if not os.path.exists(self.vectorstore_path):
            raise FileNotFoundError(
                f"Vector store not found at {self.vectorstore_path}\n"
                "Run Step 1 (process_documents.py) first!"
            )
        
        return Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=self.embeddings,
            collection_name="mining_laws"
        )
    
    def _initialize_llm(self):
        """Initialize LLM"""
        if self.llm_provider == 'huggingface':
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not api_key:
                raise ValueError("HUGGINGFACE_API_KEY not found. Set it in environment.")
            
            return HuggingFaceHub(
                repo_id='mistralai/Mistral-7B-Instruct-v0.2',
                huggingfacehub_api_token=api_key,
                model_kwargs={
                    'temperature': 0.3,
                    'max_length': 1024,
                    'max_new_tokens': 512
                }
            )
        
        elif self.llm_provider == 'groq':
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY not found. Set it in environment.")
            
            return ChatGroq(
                model='llama-3.1-70b-versatile',
                temperature=0.3,
                max_tokens=1024,
                groq_api_key=api_key
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.llm_provider}")
    
    def _create_qa_chain(self):
        """Create QA chain"""
        template = """You are an expert AI assistant for Indian mining laws and regulations.

Context from documents:
{context}

Question: {question}

Provide a detailed, accurate answer based ONLY on the context. Cite specific acts and sections.
If the information isn't in the context, say so clearly.

Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        """Process chat message"""
        try:
            # Get response from RAG
            result = self.qa_chain({"query": message})
            
            answer = result['result']
            sources = result['source_documents']
            
            # Format sources
            sources_text = self._format_sources(sources)
            
            # Combine answer and sources
            full_response = f"{answer}\n\n{sources_text}"
            
            # Save conversation
            self.conversations.append({
                'timestamp': datetime.now().isoformat(),
                'question': message,
                'answer': answer,
                'sources': [doc.metadata['source'] for doc in sources]
            })
            
            return full_response
            
        except Exception as e:
            return f" Error: {str(e)}\n\nPlease try again or rephrase your question."
    
    def _format_sources(self, sources: List) -> str:
        """Format source citations"""
        if not sources:
            return ""
        
        seen = set()
        formatted = ["---", "**📚 Sources:**"]
        
        for doc in sources:
            source = doc.metadata['source']
            if source not in seen:
                doc_type = doc.metadata.get('doc_type', 'Document')
                year = doc.metadata.get('year', 'N/A')
                formatted.append(f"- **{source}** ({doc_type}, {year})")
                seen.add(source)
        
        return "\n".join(formatted)
    
    def export_conversation(self) -> str:
        """Export conversation history"""
        if not self.conversations:
            return "No conversations to export."
        
        export_path = f"./outputs/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("./outputs", exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(self.conversations, f, indent=2)
        
        return f"Conversation exported to: {export_path}"


def create_gradio_interface(chatbot: MiningLawsChatbot):
    """Create Gradio interface"""
    
    # Custom CSS
    custom_css = """
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.HTML("""
            <div class="header">
                <h1>⛏️ Mining Laws & Regulations Assistant</h1>
                <p>Ask questions about Indian mining laws, regulations, and compliance</p>
            </div>
        """)
        
        # Main chat interface
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(
                    height=600,
                    show_label=False,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=assistant")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question about mining laws...",
                        show_label=False,
                        scale=4
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear = gr.Button("Clear Chat")
                    export = gr.Button("Export Conversation")
                
                export_output = gr.Textbox(label="Export Status", visible=False)
            
            # Sidebar with info
            with gr.Column(scale=1):
                gr.Markdown("""
                ###  Available Knowledge
                
                **Acts & Legislation:**
                - Mines Act 1952
                - MMDR Act 1957
                - Coal Mines Special Provisions Act
                
                **Regulations:**
                - Coal Mines Regulations 2017
                - Metalliferous Mines Regulations
                - Safety Standards
                
                **Reports & Circulars:**
                - DGMS Annual Reports
                - Safety Circulars
                - Compliance Guidelines
                
                ###  Tips
                
                - Be specific in your questions
                - Ask about specific acts or regulations
                - Inquire about safety requirements
                - Request compliance information
                
                ###  Example Questions
                
                - What is the Mines Act 1952?
                - Safety requirements for coal mines?
                - Minimum age for mine workers?
                - DGMS inspection procedures?
                - Environmental clearance process?
                """)
        
        # Example questions
        gr.Examples(
            examples=[
                "What are the main provisions of the Mines Act 1952?",
                "What is the minimum age for employment in mines?",
                "What are the safety requirements for underground coal mining?",
                "What environmental clearances are required for new mines?",
                "What are the penalties for illegal mining?",
                "What is DGMS and what are its responsibilities?",
                "What are the ventilation requirements in underground mines?",
                "What welfare facilities must be provided to mine workers?",
            ],
            inputs=msg,
            label="Quick Questions"
        )
        
        # Event handlers
        def respond(message, chat_history):
            response = chatbot.chat(message, chat_history)
            chat_history.append((message, response))
            return "", chat_history
        
        def export_conv():
            result = chatbot.export_conversation()
            return gr.update(value=result, visible=True)
        
        submit.click(respond, [msg, chatbot_ui], [msg, chatbot_ui])
        msg.submit(respond, [msg, chatbot_ui], [msg, chatbot_ui])
        clear.click(lambda: None, None, chatbot_ui, queue=False)
        export.click(export_conv, None, export_output)
    
    return interface


def main():
    """Main function"""
    print("\n" + "="*80)
    print("⛏️  MINING LAWS CHATBOT - WEB INTERFACE")
    print("="*80 + "\n")
    
    # Check for API key
    llm_provider = os.getenv('LLM_PROVIDER', 'huggingface')
    
    if llm_provider == 'huggingface':
        if not os.getenv('HUGGINGFACE_API_KEY'):
            print("  HUGGINGFACE_API_KEY not found!")
            print("Get your key from: https://huggingface.co/settings/tokens")
            print("Set it: export HUGGINGFACE_API_KEY='your_key'\n")
            return
    elif llm_provider == 'groq':
        if not os.getenv('GROQ_API_KEY'):
            print("  GROQ_API_KEY not found!")
            print("Get your key from: https://console.groq.com/keys")
            print("Set it: export GROQ_API_KEY='your_key'\n")
            return
    
    # Initialize chatbot
    try:
        chatbot = MiningLawsChatbot(llm_provider=llm_provider)
        
        # Create and launch interface
        interface = create_gradio_interface(chatbot)
        
        print("\n Launching web interface...")
        print("="*80)
        print(" Access the chatbot in your browser")
        print(" Public URL will be generated for sharing")
        print("  Press Ctrl+C to stop the server")
        print("="*80 + "\n")
        
        interface.launch(
            share=True,  # Creates public link
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
        
    except Exception as e:
        print(f"\n Error: {e}\n")
        print("Make sure you have:")
        print("1. Run Step 1 (process_documents.py) to create embeddings")
        print("2. Set your API key in environment")
        print("3. Installed all dependencies: pip install -r requirements.txt\n")


if __name__ == "__main__":
    main()
