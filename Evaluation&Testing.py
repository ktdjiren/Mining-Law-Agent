"""
Mining Laws RAG System - Step 4: Evaluation & Testing
=====================================================
Author: [Aman Rawat]
Description: Comprehensive evaluation and testing of the RAG system

This script evaluates:
1. Retrieval accuracy (how well it finds relevant documents)
2. Answer quality (relevance, correctness, completeness)
3. Response time and performance
4. Source citation accuracy
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# For evaluation metrics
try:
    from langchain_community.llms import HuggingFaceHub
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TestCase:
    """Test case for evaluation"""
    
    def __init__(self, question: str, expected_keywords: List[str], 
                 expected_sources: List[str] = None, category: str = "General"):
        self.question = question
        self.expected_keywords = expected_keywords
        self.expected_sources = expected_sources or []
        self.category = category


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, vectorstore_path: str = "./data/processed/mining_laws_vectorstore"):
        self.vectorstore_path = vectorstore_path
        
        print(" Initializing evaluator...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectorstore = self._load_vectorstore()
        self.llm = self._initialize_llm()
        self.qa_chain = self._create_qa_chain()
        
        self.results = []
        print(" Evaluator ready\n")
    
    def _load_vectorstore(self):
        """Load vector store"""
        if not os.path.exists(self.vectorstore_path):
            raise FileNotFoundError("Vector store not found. Run Step 1 first!")
        
        return Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=self.embeddings,
            collection_name="mining_laws"
        )
    
    def _initialize_llm(self):
        """Initialize LLM for testing"""
        api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY required for evaluation")
        
        return HuggingFaceHub(
            repo_id='mistralai/Mistral-7B-Instruct-v0.2',
            huggingfacehub_api_token=api_key,
            model_kwargs={
                'temperature': 0.1,  # Lower for more consistent evaluation
                'max_length': 1024,
                'max_new_tokens': 512
            }
        )
    
    def _create_qa_chain(self):
        """Create QA chain"""
        template = """Answer based ONLY on the context provided.

Context: {context}

Question: {question}

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
    
    def evaluate_test_case(self, test_case: TestCase) -> Dict:
        """Evaluate a single test case"""
        
        start_time = time.time()
        
        try:
            # Get response
            result = self.qa_chain({"query": test_case.question})
            response_time = time.time() - start_time
            
            answer = result['result'].lower()
            sources = result['source_documents']
            
            # Evaluate keyword presence
            keywords_found = sum(1 for kw in test_case.expected_keywords 
                               if kw.lower() in answer)
            keyword_score = keywords_found / len(test_case.expected_keywords) if test_case.expected_keywords else 0
            
            # Evaluate source relevance
            source_names = [doc.metadata['source'] for doc in sources]
            sources_found = sum(1 for src in test_case.expected_sources 
                              if any(src.lower() in s.lower() for s in source_names))
            source_score = sources_found / len(test_case.expected_sources) if test_case.expected_sources else 1.0
            
            # Calculate overall score
            overall_score = (keyword_score * 0.6 + source_score * 0.4)
            
            evaluation = {
                'question': test_case.question,
                'category': test_case.category,
                'answer': result['result'],
                'response_time': response_time,
                'keyword_score': keyword_score,
                'source_score': source_score,
                'overall_score': overall_score,
                'keywords_found': keywords_found,
                'keywords_total': len(test_case.expected_keywords),
                'sources_found': sources_found,
                'sources_total': len(test_case.expected_sources),
                'sources': source_names[:3],
                'success': True
            }
            
        except Exception as e:
            evaluation = {
                'question': test_case.question,
                'category': test_case.category,
                'error': str(e),
                'success': False
            }
        
        return evaluation
    
    def run_evaluation(self, test_cases: List[TestCase]):
        """Run evaluation on multiple test cases"""
        
        print("\n" + "="*80)
        print("🧪 STARTING EVALUATION")
        print("="*80)
        print(f"Total test cases: {len(test_cases)}\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Testing: {test_case.question[:60]}...")
            
            result = self.evaluate_test_case(test_case)
            self.results.append(result)
            
            if result['success']:
                print(f"     Score: {result['overall_score']:.2%} | Time: {result['response_time']:.2f}s")
            else:
                print(f"     Failed: {result.get('error', 'Unknown error')}")
        
        self._print_summary()
        self._save_results()
    
    def _print_summary(self):
        """Print evaluation summary"""
        successful = [r for r in self.results if r['success']]
        
        if not successful:
            print("\n❌ No successful evaluations")
            return
        
        avg_score = sum(r['overall_score'] for r in successful) / len(successful)
        avg_time = sum(r['response_time'] for r in successful) / len(successful)
        avg_keyword_score = sum(r['keyword_score'] for r in successful) / len(successful)
        avg_source_score = sum(r['source_score'] for r in successful) / len(successful)
        
        # Category breakdown
        categories = {}
        for r in successful:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r['overall_score'])
        
        print("\n" + "="*80)
        print(" EVALUATION SUMMARY")
        print("="*80)
        print(f"Total Tests:           {len(self.results)}")
        print(f"Successful:            {len(successful)}")
        print(f"Failed:                {len(self.results) - len(successful)}")
        print(f"\n Performance Metrics:")
        print(f"Average Overall Score: {avg_score:.2%}")
        print(f"Average Keyword Score: {avg_keyword_score:.2%}")
        print(f"Average Source Score:  {avg_source_score:.2%}")
        print(f"Average Response Time: {avg_time:.2f}s")
        
        print(f"\n By Category:")
        for cat, scores in categories.items():
            avg_cat_score = sum(scores) / len(scores)
            print(f"  {cat:20s}: {avg_cat_score:.2%} ({len(scores)} tests)")
        
        print("="*80 + "\n")
    
    def _save_results(self):
        """Save evaluation results"""
        output_dir = "./outputs/evaluation"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"evaluation_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'results': self.results
            }, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}\n")
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        successful = [r for r in self.results if r['success']]
        
        if not successful:
            return "No successful tests to report"
        
        avg_score = sum(r['overall_score'] for r in successful) / len(successful)
        
        report = f"""
# Mining Laws RAG System - Evaluation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Tests:** {len(self.results)}
- **Successful:** {len(successful)}
- **Failed:** {len(self.results) - len(successful)}
- **Average Score:** {avg_score:.2%}

## Test Results

"""
        
        for i, result in enumerate(successful, 1):
            report += f"""
### Test {i}: {result['category']}

**Question:** {result['question']}

**Score:** {result['overall_score']:.2%}
- Keyword Match: {result['keyword_score']:.2%}
- Source Relevance: {result['source_score']:.2%}

**Response Time:** {result['response_time']:.2f}s

**Answer:** {result['answer'][:200]}...

**Sources Used:** {', '.join(result['sources'])}

---
"""
        
        return report


def create_test_suite() -> List[TestCase]:
    """Create comprehensive test suite"""
    
    return [
        # Safety & Regulations
        TestCase(
            question="What are the safety requirements for coal mines?",
            expected_keywords=["safety", "coal", "regulation", "ventilation", "protection"],
            expected_sources=["COAL MINES REGULATION", "safety"],
            category="Safety"
        ),
        TestCase(
            question="What is the minimum age for employment in mines?",
            expected_keywords=["18", "age", "employment", "minor", "years"],
            expected_sources=["Mines Act", "Employment"],
            category="Labour"
        ),
        TestCase(
            question="What are the ventilation requirements in underground mines?",
            expected_keywords=["ventilation", "underground", "air", "cubic", "requirement"],
            expected_sources=["regulation", "safety"],
            category="Safety"
        ),
        
        # Acts & Legislation
        TestCase(
            question="What is the Mines Act 1952?",
            expected_keywords=["1952", "act", "mines", "regulation", "safety"],
            expected_sources=["Mines Act"],
            category="Legislation"
        ),
        TestCase(
            question="What is MMDR Act?",
            expected_keywords=["mineral", "development", "regulation", "act"],
            expected_sources=["MMDR"],
            category="Legislation"
        ),
        
        # Compliance & Penalties
        TestCase(
            question="What are the penalties for illegal mining?",
            expected_keywords=["penalty", "fine", "imprisonment", "illegal", "violation"],
            expected_sources=["Act", "regulation"],
            category="Compliance"
        ),
        TestCase(
            question="What environmental clearances are required for mining?",
            expected_keywords=["environmental", "clearance", "approval", "permission"],
            expected_sources=["environment", "forest"],
            category="Compliance"
        ),
        
        # DGMS
        TestCase(
            question="What is DGMS and its role?",
            expected_keywords=["DGMS", "inspection", "safety", "mines", "director"],
            expected_sources=["DGMS"],
            category="Administration"
        ),
        TestCase(
            question="What are DGMS inspection procedures?",
            expected_keywords=["inspection", "DGMS", "procedure", "safety", "compliance"],
            expected_sources=["DGMS"],
            category="Administration"
        ),
        
        # Worker Welfare
        TestCase(
            question="What welfare facilities must be provided to mine workers?",
            expected_keywords=["welfare", "facility", "worker", "provision", "amenity"],
            expected_sources=["Mines Act", "welfare"],
            category="Welfare"
        ),
        TestCase(
            question="What are the provisions for women working in mines?",
            expected_keywords=["women", "employment", "provision", "work", "hour"],
            expected_sources=["Employment", "Women"],
            category="Labour"
        ),
        
        # Technical
        TestCase(
            question="What are the blasting regulations in mines?",
            expected_keywords=["blast", "explosive", "regulation", "safety", "permission"],
            expected_sources=["regulation", "safety"],
            category="Technical"
        ),
        TestCase(
            question="What are the requirements for mine planning?",
            expected_keywords=["plan", "mining", "requirement", "approval", "design"],
            expected_sources=["regulation", "Act"],
            category="Technical"
        ),
    ]


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("  MINING LAWS RAG SYSTEM - EVALUATION & TESTING")
    print("="*80 + "\n")
    
    # Check API key
    if not os.getenv('HUGGINGFACE_API_KEY'):
        print(" HUGGINGFACE_API_KEY not found!")
        print("Set it: export HUGGINGFACE_API_KEY='your_key'\n")
        return
    
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator()
        
        # Create test suite
        test_cases = create_test_suite()
        
        print(f" Created {len(test_cases)} test cases")
        print(f" Categories: {len(set(tc.category for tc in test_cases))}")
        
        # Run evaluation
        evaluator.run_evaluation(test_cases)
        
        # Generate report
        report = evaluator.generate_report()
        report_file = "./outputs/evaluation/evaluation_report.md"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f" Report saved to: {report_file}")
        
    except Exception as e:
        print(f"\n Error: {e}\n")


if __name__ == "__main__":
    main()
