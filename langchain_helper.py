from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()


def create_vector_db(pdf_path)->FAISS:
    loader = PyPDFLoader(file_path=pdf_path, extract_images=False)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    db = FAISS.from_documents(texts, embeddings)
    return db

def get_response(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    text = " ".join([d.page_content for d in docs])
    llm = OpenAI()
    prompt = PromptTemplate(
        input_variables= ["question","docs"],
        template= """You are a helpful assistant that can answer questions about a document.
        Answer the following question: {question}
        By searching the following document: {docs} 
        Only use the factual information from the document to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answer should be verbose and detailed.
        """
    )
    chain = LLMChain(llm =llm, prompt = prompt, output_key = "ans")
    response = chain.invoke({'question': query, 'docs': text})
    response['ans'] = response['ans'].replace("\n"," ")
    return response['ans']
