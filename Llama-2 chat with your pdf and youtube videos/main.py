from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
import sys
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
import chromadb
from langchain.document_loaders import YoutubeLoader


#**Step 1: Load the PDF File from Data Path****
loader = PyPDFLoader("C:/Users/HP/Downloads/Staj_Defteri_Ornegi (5).pdf")
pages = loader.load()


print(len(pages))
###***for youtube videos activate these lines
# loader = YoutubeLoader.from_youtube_url(
#     "https://www.youtube.com/watch?v=jGwO_UgTS7I", add_video_info=True
# )
# docs_youtube=loader.load()
#docs_video = r_splitter.split_documents(docs_youtube)

#***Step 2: Split Text into Chunks***

text_splitter=RecursiveCharacterTextSplitter(
                                             chunk_size=700,
                                             chunk_overlap=70,
                                             separators=["\n\n", "\n", "(?<=\. )", " ", ""])


text_chunks=text_splitter.split_documents(pages)

print(len(text_chunks))
#**Step 3: Load the Embedding Model***


embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':'cpu'})



#**Step 4: Convert the Text Chunks into Embeddings and Create a FAISS Vector Store***
persist_directory='D:\Test_embad'

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory=persist_directory
) 

# if the embadding has been done before by activating these lines you could read it directly
""" vectordb=Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
) """


print(vectordb._collection.count())



llm=CTransformers(model="models\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':128,
                          'temperature':0.01})


template="""Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])




chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=vectordb.as_retriever(),
                                   return_source_documents=True,
                                   chain_type_kwargs={'prompt': qa_prompt})




while True:
    user_input=input(f"prompt:")
    if user_input=='prompt:exit':
        print('Exiting')
        sys.exit()
    if user_input=='"prompt:"':
        continue
    result=chain({'query':user_input})
    print(f"Answer:{result['result']}")

print("done")