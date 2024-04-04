"""!pip install -q llama-index
!pip install -q openai
!pip install -q transformers
!pip install -q accelerate
"""

import os
os["OPENAI_API_KEY"] = "YOUR_API_KEY"

from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from IPython.display import Markdown, display

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up? ")

display(Markdown(f"<b>{response}</b>"))

"""### Storing and Loading the Index"""

index.storage_context.persist()

from llama_index import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context=storage_context)



"""### How to Customize"""

from llama_index import ServiceContext, set_global_service_context

# define LLM: https://gpt-index.readthedocs.io/en/latest/core_modules/model_modules/llms/usage_custom.html
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)

# configure service context
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=800, chunk_overlap=20)
# set_global_service_context(service_context)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# from llama_index.llms import PaLM
# service_context = ServiceContext.from_defaults(llm=PaLM())

query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("What did the author do growing up?")
response.print_response_stream()



query_engine = index.as_chat_engine()
response = query_engine.chat("What is this document about?")
display(Markdown(f"<b>{response}</b>"))



"""#### Using a HuggingFace LLM

This will NOT work on the Free Google Colab. You will need colab Pro. Video on the quantized models is coming soon.....
"""

from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

import torch
from llama_index.llms import HuggingFaceLLM
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
    model_name="StabilityAI/stablelm-tuned-alpha-3b",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)























