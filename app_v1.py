"""
app_v1.py:  v1 of the app

1. allows the user to upload multiple files at runtime (upto a maximum, that is set via a paremeter)
    a. user is told about the max; cl max is 10 files
    b. file type can either be .txt or .pdf
    c. see the use of the accept option in the cl.AskFileMessage object: 
        set this as: accept=["text/plain", "application/pdf"]
2. uploaded files are processed on start:
    if pdf, then first convert to text: i.e., the collection of uploaded files is converted to text if needed
    text is split into chunks using langchain RecursiveCharacterTextSplitter util - 
        options TBD
3. uses openai embeddings (specifically, text-embedding-3-small embeddings) to convert chunks into embeddings
    note - the choice of embeddings model can be controlled via optional parameters to be passed in when
            the vector db is instantiated
4. save these embeddings in a vector db
    hope to use Chroma or Qdrant
    NOTES here...

5. instantiate the RAQA (retrieval augmented qa) pipeline
    use LangChain
    chat memory needed

    requires instantiation of an openai client session and the vector db
    in addition, of course we need to set up system and user prompts
        here this is set up using aimakerspace.openai.utils prompts.py module classes
6. the cl.on_message decorator wraps the main function
    this function 
        a. receives the query that the user types in
        b. runs the RAQA pipeline
        c. sends results back to UI for dislay

Additional Notes:
a. note the use of async functions and await async syntax throughout the module here!
b. note the use of yield rather than return in certain key functions
c. note the use of streaming capabilities when needed
d. the use of the python tempfile module when the user input text file is processed
    (i) NamedTemporaryFile 
    (ii) with options to persist the storage of the temp file

"""

import os
from typing import List
from dotenv import load_dotenv
import tempfile
import getpass
from uuid import uuid4
import tempfile

# chainlit imports
import chainlit as cl
from chainlit.types import AskFileResponse

# langchain imports
# document loader
from langchain_community.document_loaders import TextLoader, PyPDFLoader
# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# embeddings model to embed each chunk of text in doc
from langchain_openai import OpenAIEmbeddings
# vector store
from langchain_chroma import Chroma
# llm for text generation using prompt plus retrieved context plus query
from langchain_openai import ChatOpenAI
# templates to create custom prompts
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# chains 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# LCEL Runnable Passthrough
from langchain_core.runnables import RunnablePassthrough
# to parse output from llm
from langchain_core.output_parsers import StrOutputParser

from langchain.docstore.document import Document

# aimakerspace imports
# from aimakerspace.text_utils import CharacterTextSplitter, TextFileLoader
# from aimakerspace.openai_utils.prompts import (
#     UserRolePrompt,
#     SystemRolePrompt,
#     AssistantRolePrompt,
# )
# from aimakerspace.openai_utils.embedding import EmbeddingModel
# from aimakerspace.vectordatabase import VectorDatabase
# from aimakerspace.openai_utils.chatmodel import ChatOpenAI


# use getpass to load api keys at runtime
# alternative is to use load_dotenv() 
# or, load secrets (eg on hf)
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
load_dotenv()


# parameters to manage number of files imported via cl onstart
max_file_count = 2

# parameters to manage splitting
chunk_kwargs = {
    'chunk_size': 1000,
    'chunk_overlap': 100
}

# openai embeddings model parameters
openai_embed_kwargs = {
    'model': 'text-embedding-3-small',
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # 'dimensions': 1024
}

# chat model parameters
chat_model_name = {
    'model_name': 'gpt-4o-mini'
}

openai_embed_kwargs = {
    'model': 'text-embedding-3-large',
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # 'dimensions': 1024
}

retriever_kwargs = {
    'search_type': 'similarity',
    'search_kwargs': {
        'k': 20
    }
}

system_prompt = (
        """You are an assistant for question-answering tasks.
        You will be given political speeches by leading politicians and
        will be asked questions based on these speeches.

        Use the following pieces of retrieved context to answer 
        the question. 
        
        You must answer the question only based on the context provided.
        
        If you don't know the answer or if the context does not provide sufficient information, 
        then say that you don't know. 
        
        Think through your answer step-by-step.

        \n\n

        Context:
        {context}
        """
    )

custom_rag_template = """\
You are an assistant for question-answering tasks.
You will be given political speeches by leading politicians and
will be asked questions based on these speeches.

Use the following pieces of retrieved context to answer 
the question. 

You must answer the question only based on the context provided.

If you don't know the answer or if the context does not provide sufficient information, 
then say that you don't know. 

Think through your answer step-by-step.

Context:
{context}

Question: 
{question}

Helpful Answer:
"""


class RetrievalAugmentedQAPipelineWithLangchain:
    def __init__(self, 
                 list_of_files: List[AskFileResponse],
                 chunk_kwargs,
                 embed_kwargs,
                 retriever_kwargs,
                 system_prompt,
                 chat_model_name,
                 system_prompt_template=None,
                 use_lcel_for_chain=False):
        self.list_of_files = list_of_files
        self.chunk_kwargs = chunk_kwargs
        self.embed_kwargs = embed_kwargs
        self.retriever_kwargs = retriever_kwargs
        self.system_prompt = system_prompt
        self.chat_model_name = chat_model_name
        self.system_prompt_template = system_prompt_template
        self.use_lcel_for_chain = use_lcel_for_chain

        self._setup_text_splitter()
        self._setup_embeddings()
        self._setup_llm()
        return
    
    def _setup_text_splitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            add_start_index=True,
            **self.chunk_kwargs
        )
        return self
    
    def _setup_embeddings(self):
        self.embeddings = OpenAIEmbeddings(**self.embed_kwargs)
        return self
    
    def _setup_llm(self):
        self.llm = ChatOpenAI(**self.chat_model_name)
        return self
    
    def format_docs(self):
        return "\n\n".join(doc.page_content for doc in self.all_splits)

    def process_all_uploaded_files(self):
        self.all_splits = []
        for file in self.list_of_files:
            # set loader depending on type of file
            if file.type == "text/plain":
                Loader = TextLoader
            elif file.type == "application/pdf":
                Loader = PyPDFLoader
            
            import tempfile
            # make temporary copy of file and split into chunks
            with tempfile.NamedTemporaryFile() as tempfile:
                tempfile.write(file.content)
                loader = Loader(tempfile.name)
                documents = loader.load()
                splits = self.text_splitter.split_documents(documents)
                for i, split in enumerate(splits):
                    split.metadata["source"] = f"{file.name}_source_{i}"
                self.all_splits.extend(splits)
        return self

    def make_chroma_vector_db(self):
        # initialize vector store
        vectorstore = Chroma.from_documents(documents=self.all_splits, embedding=self.embeddings)
        self.retriever = vectorstore.as_retriever(**self.retriever_kwargs)
        return self
    
    def setup_chat_prompt_from_messages(self):
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )
        return self
    
    def qa_chain(self):
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.chat_prompt)
        return self
    
    def rag_chain(self):
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
        return self

    def setup_prompt_from_template(self):
        self.prompt_from_template = PromptTemplate.from_template(self.system_prompt_template)
        return self

    def lcel_rag_chain(self):
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt_from_template
            | self.llm
            | StrOutputParser()
        )
    
    def make_raqa_pipeline(self):
        # load all docs uploaded by user, convert into text if needed and split into chunks
        self.process_all_uploaded_files()

        # load all splits and embeddings into vector db
        self.make_chroma_vector_db()

        if self.use_lcel_for_chain is False:
            self.setup_chat_prompt_from_messages()
            self.qa_chain()
            self.rag_chain()
        else:
            self.setup_prompt_from_template()
            self.lcel_rag_chain()
        return self.rag_chain


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload one or more files
    user_input_files = []
    filecount = 0
    user_signals_done = False
    while filecount < max_file_count and user_signals_done is False:
    # while files == None:
        files = await cl.AskFileMessage(
            content=f"Please upload {max_file_count} text files or pdf documents to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            max_files=2,
            timeout=180,
        ).send()

        if files:
            user_input_files.append(files[0])
            filecount += 1
        else:
            user_signals_done = True

    user_input_file_names = [x.name for x in user_input_files]

    msg = cl.Message(
        content=f"Processing `{user_input_file_names}`...", disable_human_feedback=True
    )
    await msg.send()

    # instantiate raqa pipeline object
    qabot = RetrievalAugmentedQAPipelineWithLangchain(
        list_of_files=user_input_files,
        chunk_kwargs=chunk_kwargs,
        embed_kwargs=openai_embed_kwargs,
        retriever_kwargs=retriever_kwargs,
        system_prompt=system_prompt,
        chat_model_name=chat_model_name
    )

    raqa_chain = qabot.make_raqa_pipeline()

    # Let the user know that the system is ready
    msg.content = f"Processing `{user_input_file_names}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("raqa_chain", raqa_chain)


@cl.on_message
async def main(message):
    raqa_chain = cl.user_session.get("raqa_chain")

    msg = cl.Message(content="")

    # result = await raqa_chain.invoke({"input": message.content})
    result = await cl.make_async(raqa_chain.invoke)({"input": message.content})

    # async for stream_resp in result["answer"]:
    for stream_resp in result["answer"]:
        await msg.stream_token(stream_resp)

    await msg.send()
