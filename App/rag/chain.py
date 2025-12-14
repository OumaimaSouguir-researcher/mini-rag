"""
RAG chain assembly module.
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Custom prompt template for RAG
RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided context.

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""


def build_chain(llm, store, chain_type: str = "stuff"):
    """
    Build RAG chain from LLM and vector store.
    
    Args:
        llm: Language model instance
        store: Vector store instance
        chain_type: Type of chain ("stuff", "map_reduce", "refine", "map_rerank")
        
    Returns:
        RetrievalQA chain instance
    """
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print(f"Built RAG chain with chain_type={chain_type}")
    return chain


def build_custom_chain(llm, retriever):
    """
    Build a custom RAG chain with more control.
    
    Args:
        llm: Language model instance
        retriever: Retriever instance
        
    Returns:
        Custom chain instance
    """
    from langchain.chains.question_answering import load_qa_chain
    
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    
    return qa_chain