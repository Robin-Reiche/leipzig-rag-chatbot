"""The RAG engine: indexing, retrieval and the conversational chat engine.

This assembles the parts from ``model_loader`` into a working Retrieval-
Augmented Generation pipeline:

1. Indexing  - read the documents in ``data/``, split them into chunks and
   embed them into a persistent vector store (built once, cached on disk).
2. Retrieval - find the most relevant chunks for a question.
3. Generation - hand those chunks plus the question to the LLM, which answers
   grounded in the sources.

Two engines live here: a lean high-level chat engine used by the command-line
mode, and an explicit streaming RagChatbot used by the web UI.
"""

from collections.abc import Iterator

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import Document, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import (
    CHAT_MEMORY_TOKEN_LIMIT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONDENSE_ENABLED,
    CONDENSE_PROMPT_TEMPLATE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    RERANK_ENABLED,
    RERANK_RETRIEVE_TOP_K,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
)
from src.model_loader import get_embedding_model, initialise_llm


def _create_new_vector_store(
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """Creates, saves and returns a new vector store from the documents."""

    print(f"Building a new vector store from documents in '{DATA_PATH}' ...")

    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DATA_PATH.as_posix()
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Run 'python prepare_data.py' "
            "first to download the Leipzig knowledge base."
        )

    # Strip the provenance header (Titel/Quelle/Lizenz/Abgerufen + a dashed
    # line) that prepare_data.py prepends, so it does not pollute embeddings or
    # snippets. The header stays in the files for attribution and is read
    # separately by the server to build the source links.
    separator = "-" * 60
    for document in documents:
        if document.text.lstrip().startswith("Titel:") and separator in document.text:
            document.set_content(document.text.split(separator, 1)[1].lstrip())

    # Break long documents into smaller, overlapping chunks.
    text_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Embed every chunk and store the vectors.
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model,
        show_progress=True,
    )

    # Persist to disk so subsequent runs load instantly.
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print(f"Vector store created from {len(documents)} documents and saved.")
    return index


def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """Loads the vector store from disk, or builds it if it does not exist."""

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk ...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model,
        )

    return _create_new_vector_store(embed_model)


def get_chat_engine(
    llm: LLM,
    embed_model: HuggingFaceEmbedding,
) -> BaseChatEngine:
    """Assembles the conversational RAG chat engine.

    Uses "context" mode: for every question it retrieves the most relevant
    chunks and injects them into the system prompt, so each answer is grounded
    in the knowledge base. A memory buffer keeps the conversation coherent.
    """

    vector_index: VectorStoreIndex = get_vector_store(embed_model)
    memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )

    return vector_index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        llm=llm,
        system_prompt=LLM_SYSTEM_PROMPT,
        similarity_top_k=SIMILARITY_TOP_K,
    )


def build_rag_chatbot() -> BaseChatEngine:
    """Convenience builder: initialises every component and returns the engine.

    Used by both the command-line loop and the web server.
    """

    print("Initialising models ...")
    llm: LLM = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()
    chat_engine: BaseChatEngine = get_chat_engine(llm=llm, embed_model=embed_model)
    print("RAG chatbot ready.")
    return chat_engine


class RagChatbot:
    """A streaming RAG chatbot, built for the web UI.

    Unlike LlamaIndex's high-level chat engine (which buffers the answer before
    returning it), this exposes the pipeline as explicit, observable steps so
    the front-end can stream them live: retrieve sources, then generate the
    answer token by token. It keeps a memory buffer so the conversation stays
    coherent across turns.
    """

    def __init__(
        self,
        llm: LLM,
        retriever: BaseRetriever,
        system_prompt: str,
        memory_token_limit: int,
        reranker: SentenceTransformerRerank | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.system_prompt = system_prompt
        self.memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(
            token_limit=memory_token_limit
        )

    def _condense_question(self, question: str) -> str:
        """Rewrites a follow-up into a standalone query using the chat history.

        Returns the question unchanged on the first turn (no history) or if
        condensing is disabled or fails. The rewrite is an optimisation, so a
        failed call falls back to the original question instead of breaking the
        turn.
        """

        if not CONDENSE_ENABLED:
            return question
        history = self.memory.get()
        if not history:
            return question

        history_text = "\n".join(
            f"{message.role.value}: {message.content}" for message in history
        )
        prompt = CONDENSE_PROMPT_TEMPLATE.format(
            chat_history=history_text, question=question
        )
        try:
            condensed = str(self.llm.complete(prompt)).strip()
        except Exception as error:  # condensing is optional, never fatal
            print(f"    ! condense failed ({error}), using original question")
            return question
        return condensed or question

    def retrieve(self, question: str) -> list[NodeWithScore]:
        """Step 1+2: condense the question, retrieve broadly, then rerank.

        Follow-ups are first condensed against the chat history into a stand-
        alone query (one extra LLM call, only when history exists). The vector
        store then returns a broad set of candidates, and the cross-encoder
        reranker re-scores them and keeps only the most relevant few.
        """

        search_query = self._condense_question(question)
        nodes = self.retriever.retrieve(search_query)
        if self.reranker is not None:
            nodes = self.reranker.postprocess_nodes(
                nodes, query_str=search_query
            )
        return nodes

    def _system_message(self, nodes: list[NodeWithScore]) -> ChatMessage:
        """Builds the system prompt with the retrieved context injected."""

        context = "\n\n".join(
            f"[Quelle: {node.node.metadata.get('file_name', 'unbekannt')}]\n"
            f"{node.node.get_content()}"
            for node in nodes
        )
        content = (
            f"{self.system_prompt}\n\n"
            "Nutze ausschließlich den folgenden Kontext aus der Wissensbasis, "
            "um die Frage zu beantworten:\n"
            "--- KONTEXT ---\n"
            f"{context}\n"
            "--- ENDE KONTEXT ---"
        )
        return ChatMessage(role=MessageRole.SYSTEM, content=content)

    def stream_answer(
        self,
        question: str,
        nodes: list[NodeWithScore],
    ) -> Iterator[str]:
        """Step 3: generate the answer, yielding it token by token."""

        user_message = ChatMessage(role=MessageRole.USER, content=question)
        system_message = self._system_message(nodes)

        # Count the system + context tokens so the memory budget accounts for
        # them. Otherwise history pruning ignores the largest part of the
        # prompt (the injected chunks) and the prompt can overflow the window.
        initial_tokens = len(
            self.memory.tokenizer_fn(str(system_message.content))
        )
        history = self.memory.get(initial_token_count=initial_tokens)
        messages = [system_message] + history + [user_message]

        answer = ""
        for chunk in self.llm.stream_chat(messages):
            delta = chunk.delta or ""
            if delta:
                answer += delta
                yield delta

        # Commit the turn to memory only after a successful generation, so a
        # failed or empty stream never leaves a dangling user message that
        # would corrupt the next turn's history.
        self.memory.put(user_message)
        self.memory.put(
            ChatMessage(role=MessageRole.ASSISTANT, content=answer)
        )

    def reset(self) -> None:
        """Clears the conversation memory."""

        self.memory.reset()

    def set_llm(self, llm: LLM) -> None:
        """Swaps the generator model used from the next answer onward.

        Only the generator changes; the retriever, reranker and memory stay, so
        the conversation continues seamlessly with the new model.
        """

        self.llm = llm


def build_streaming_chatbot() -> RagChatbot:
    """Builds the streaming RAG chatbot used by the web server."""

    print("Initialising models ...")
    llm: LLM = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()
    vector_index: VectorStoreIndex = get_vector_store(embed_model)

    # With reranking on, retrieve a broad candidate set and let the reranker
    # narrow it down. Without it, retrieve the final count directly.
    retrieve_top_k = RERANK_RETRIEVE_TOP_K if RERANK_ENABLED else SIMILARITY_TOP_K
    retriever: BaseRetriever = vector_index.as_retriever(
        similarity_top_k=retrieve_top_k
    )

    reranker: SentenceTransformerRerank | None = None
    if RERANK_ENABLED:
        print(f"Loading reranker '{RERANKER_MODEL_NAME}' ...")
        reranker = SentenceTransformerRerank(
            model=RERANKER_MODEL_NAME, top_n=RERANKER_TOP_N
        )

    print("RAG chatbot ready.")
    return RagChatbot(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        system_prompt=LLM_SYSTEM_PROMPT,
        memory_token_limit=CHAT_MEMORY_TOKEN_LIMIT,
    )


def main_chat_loop() -> None:
    """Runs the RAG chatbot as an interactive command-line REPL."""

    chat_engine: BaseChatEngine = build_rag_chatbot()
    print("\n--- Leipzig-Experte (CLI) --- type 'exit' to quit ---\n")
    chat_engine.chat_repl()
