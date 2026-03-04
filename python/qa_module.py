"""
Q&A Module - Answers questions based ONLY on user-provided PDF data.
No external knowledge or third-party data is used.
"""

import os
import hashlib
import json
import time
from typing import Optional, List
from PyPDF2 import PdfReader
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document



class CustomQAModule:
    def __init__(self, pdf_path: str, clear_cache: bool = False):
        self.pdf_path = pdf_path
        self.documents: List[Document] = []
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.qa_chain = None
        self.answer_cache = {}  
        self.cache_file = "data/answer_cache.json"
        
        
        if clear_cache:
            self._clear_all_cache()

        
        self._load_pdf()
        self._create_vector_store()
        self._setup_qa_chain()
        self._load_cache()
    
    def _clear_all_cache(self):
        """Clear cache and vector DB."""
        import shutil
        persist_path = "data/chroma_db"
        if os.path.exists(persist_path):
            try:
                shutil.rmtree(persist_path)
                print(f"Cleared vector DB at {persist_path}")
            except:
                pass
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                print(f"Cleared cache at {self.cache_file}")
            except:
                pass

    def _load_cache(self):
        """Load cached answers from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.answer_cache = json.load(f)
                    print(f"Loaded {len(self.answer_cache)} cached answers")
            except:
                self.answer_cache = {}

    def _save_cache(self):
        """Save answers to cache file."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.answer_cache, f)

    def _get_question_hash(self, question: str) -> str:
        """Generate a hash for a question."""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()

    def _format_answer(self, answer: str) -> str:
        """Format answer with proper line breaks - each sentence on new line."""
        
        answer = answer.replace('\\n', ' ').replace('\\r', ' ')
        answer = answer.replace('/n', ' ').replace('/n1', ' ').replace('/n2', ' ')
      
        answer = answer.strip()
       
        import re
       
        sentences = re.split(r'(?<=[.!?])\s+', answer)
     
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return '\n'.join(sentences)

 
    def _load_pdf(self):
        print(f"Loading PDF: {self.pdf_path}")

        reader = PdfReader(self.pdf_path)
        docs = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": page_num + 1}
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  
            chunk_overlap=40,  
            separators=["\n\n", "\n", " ", ""]
        )

        self.documents = splitter.split_documents(docs)
        print(f"Loaded {len(self.documents)} chunks from PDF")

   
    def _create_vector_store(self):
        print("Creating vector embeddings...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  
        )

        
        persist_path = "data/chroma_db"

       
        if os.path.exists(persist_path):
            print("Loading existing vector DB...")
            self.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_path
            )
        else:
            print("Creating new vector DB...")
            self.vector_store = Chroma.from_documents(
                documents=self.documents,
                embedding=embeddings,
                persist_directory=persist_path
            )

        print("Vector store ready")

   
    def _setup_qa_chain(self):
        assert self.vector_store is not None

        llm = OllamaLLM(
            model="mistral",
            temperature=0,
            base_url="http://localhost:11434",
            num_predict=150,  
            top_k=20,  
            top_p=0.95,
            repeat_penalty=1.0
        )

        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  
        )
        

        prompt = self._get_custom_prompt()

        def format_docs(docs):
            return "\n\n".join(
                [d.page_content for d in docs]
            )

       
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": lambda x: x
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("Q&A chain ready")

   
    def _get_custom_prompt(self):
        template = """You are a knowledgeable assistant answering questions based ONLY on the provided document.

Your task:
1. Read all the provided context carefully
2. Search for ANY mention of the topic/concept mentioned in the question
3. Look for related terms, definitions, explanations, and examples
4. Provide the answer DIRECTLY from the context

CRITICAL RULES:
- ALWAYS answer if the topic/concept appears ANYWHERE in the context, even if indirectly
- Combine information from multiple parts of the context if needed
- Do NOT refuse to answer - search thoroughly first
- Only say "I don't have this information" if the topic is COMPLETELY absent from context

Document Content:
{context}

User Question: {question}

Answer directly from the provided content:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def ask(self, question: str) -> dict:
        assert self.retriever is not None
        assert self.qa_chain is not None

        start_time = time.time()

       
        q_hash = self._get_question_hash(question)
        if q_hash in self.answer_cache:
            elapsed = time.time() - start_time
            print(f"[CACHE HIT] Returned in {elapsed:.2f}s")
            return {"answer": self.answer_cache[q_hash], "time": elapsed}

        try:
            
            relevant_docs = self.retriever.invoke(question)
            
          
            if len(relevant_docs) < 2:
                
                key_terms = question.split()
                for term in key_terms:
                    if len(term) > 3:  
                        additional_docs = self.retriever.invoke(term)
                        relevant_docs.extend(additional_docs)
                
               
                seen = set()
                unique_docs = []
                for doc in relevant_docs:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen:
                        seen.add(content_hash)
                        unique_docs.append(doc)
                relevant_docs = unique_docs[:10]  
            
           
            answer = self.qa_chain.invoke(question)
           
            answer = self._format_answer(answer)
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {"answer": "Unable to generate answer. Please try again.", "time": time.time() - start_time}

       
        self.answer_cache[q_hash] = answer
        self._save_cache()

        elapsed = time.time() - start_time
        print(f"[NEW] Answer generated in {elapsed:.2f}s")
        return {
            "answer": answer,
            "time": elapsed
        }


if __name__ == "__main__":

    pdf_path = "data/exchange.pdf"

    if not os.path.exists(pdf_path):
        print(f"PDF not found at {pdf_path}")
        exit(1)

    qa = CustomQAModule(pdf_path)

    while True:
        q = input("\nAsk a question (or type exit): ")
        if q.lower() == "exit":
            break

        result = qa.ask(q)
        print("\nANSWER:", result["answer"])
