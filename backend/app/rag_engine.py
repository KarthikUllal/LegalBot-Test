# backend/app/rag_engine.py
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging
from .config import settings
from .provider_client import get_best_provider, ProviderClient
from .vector_store import VectorStore
from .ingestion import split_into_chunks, load_pdf_text
from .schemas import ChatResponse, SourceItem
from .translation import translator
import re
from datetime import datetime

from langchain.memory import ConversationBufferMemory  # NEW: For conversation history
from langchain.memory import ConversationBufferWindowMemory
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)  # For query embedding

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough

logger = logging.getLogger(__name__)


# Main RAG engine class - handles document ingestion, retrieval, and generation
class RAGEngine:
    def __init__(self, provider: ProviderClient = None):
        self.provider = provider or get_best_provider()
        self.vstore = VectorStore()
        self.memory = ConversationBufferWindowMemory(
            k=8,  # Keep last 8 exchanges (same as your old limit)
            memory_key="chat_history",
            return_messages=True,
        )
        self.conversation_transcripts = (
            {}
        )  # session_id -> full lawyer-style transcript (keep for downloads)

        self._setup_prompts()
        self._setup_chain()

        logger.info(
            f"RAG Engine initialized with provider: {type(self.provider).__name__}"
        )

    def _setup_prompts(self):
        # Clean context formatting
        self.context_prompt = PromptTemplate(
            input_variables=["context"],
            template="""Relevant Legal Provisions from Indian Laws:

    {context}

    Use only the above provisions to answer accurately and naturally.""",
        )

        # FINAL PERFECT PROMPT — handles greetings, new questions, and follow-ups intelligently
        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""You are "Nyaya Mitra" — a friendly, professional, and trusted AI legal assistant helping people understand Indian laws in simple, clear language.

    Current User Message: "{question}"

    Previous Conversation (if any):
    {chat_history}

    Relevant Legal Provisions:
    {context}

    **Critical Instructions — Follow Exactly:**
    - If the user message is a greeting (hello, hi, namaste, bye, thanks, how are you, etc.) or very short — respond warmly and naturally. Treat it as a fresh start.
    - Only use the previous conversation if the current message clearly refers to it (e.g., "what is the punishment?", "how to file complaint?", "tell me more", "and then?").
    - If the message is a new legal question or unrelated — ignore chat history and answer fresh.
    - Never force connection to previous topic.
    - For legal answers — use ONLY the legal provisions above.
    - Speak directly and naturally, like a trusted lawyer.
    - Refer to laws clearly:
    - "Under Section 318 of the Bharatiya Nyaya Sanhita (BNS)..."
    - "The punishment is..."
    - "You should first go to..."
    - Always prefer current law (BNS, BNSS, BSA) over old IPC.
    - BNS fully replaced IPC on 1 July 2024.
    - Be concise but complete (4–8 sentences).
    - Use simple, supportive language.
    - If provisions are insufficient — say honestly: "I don't have full details from the legal texts, but generally..."

    Now respond in a warm, professional, and accurate way:""",
        )

    def _setup_chain(self):
        pass

    def _clean_response(self, response: str) -> str:
        response = re.sub(r"\n\s*\n", "\n\n", response)

        sections = [
            "Overview",
            "Key Definitions",
            "Legal Provisions",
            "Punishments & Penalties",
            "Important Points",
            "Legal References:",
        ]

        for section in sections:
            if section in response and not response.startswith(section):
                response = response.replace(section, f"\n\n{section}")

        return response.strip()

    def _format_context(
        self, docs: List[str], metas: List[dict], ids: List[str], dists: List[float]
    ) -> str:
        if not docs:
            return "No specific legal provisions found for this query."

        parts = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            act = meta.get("act", "Indian Law")
            section = meta.get("section", "")
            section_text = (
                f"Section {section}"
                if section and section != "None"
                else "Relevant Provision"
            )

            parts.append(f"From {act} ({section_text}):\n{doc.strip()}\n")

        return "\n".join(parts)

    def ingest_text(self, doc_id: str, text: str, metadata: dict = None) -> bool:
        try:
            logger.info(f"Ingesting document: {doc_id}")
            chunks = split_into_chunks(text)
            logger.info(f"Split into {len(chunks)} chunks")

            from .ingestion import enrich_metadata_with_section

            base_meta = {
                **(metadata or {}),
                "doc_id": doc_id,
                "act": metadata.get("act", doc_id) if metadata else doc_id,
            }
            metadatas = enrich_metadata_with_section(chunks, base_meta)

            ids = [f"{doc_id}__{i}" for i in range(len(chunks))]

            # NEW: Batch adding to avoid ChromaDB limit
            batch_size = 500  # Safe size (adjust up to ~4000 if needed)
            for i in range(0, len(chunks), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_docs = chunks[i : i + batch_size]
                batch_metas = metadatas[i : i + batch_size]

                logger.info(
                    f"Adding batch {i//batch_size + 1}: {len(batch_docs)} chunks"
                )
                self.vstore.collection.add(
                    ids=batch_ids, documents=batch_docs, metadatas=batch_metas
                )

            logger.info(
                f"Successfully ingested {len(chunks)} chunks from {doc_id} in batches"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to ingest {doc_id}: {e}", exc_info=True)
            return False

    def ingest_file(
        self, file_path: str, doc_id: str = None, act_name: str = None
    ) -> bool:
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            doc_id = doc_id or path.stem
            logger.info(f"Processing file: {path.name} as {doc_id}")

            text = load_pdf_text(path)
            if not text:
                raise ValueError(
                    "No text extracted from PDF. File might be scanned or corrupted."
                )

            metadata = {
                "act": act_name or doc_id,
                "source_file": path.name,
                "source_type": "pdf",
            }

            return self.ingest_text(doc_id, text, metadata)

        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
            return False

    def retrieve(self, query: str, k: int = 6) -> Dict:
        try:
            logger.info(f"Retrieving for: '{query}'")

            # Use Chroma's embedding function (same model: all-MiniLM-L6-v2)
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            # ✅ IMPROVED LAW CODE DETECTION
            requested_law = None
            query_lower = query.lower()

            if any(
                term in query_lower
                for term in [
                    "bns",
                    "bharatiya nyaya sanhita",
                    "bharatiya nyaya",
                    "nyaya sanhita",
                    "भरतीया न्याय संहिता",
                ]
            ):
                requested_law = "BNS"
            elif any(
                term in query_lower
                for term in [
                    "bnss",
                    "bharatiya nagarik suraksha sanhita",
                    "nagarik suraksha",
                ]
            ):
                requested_law = "BNSS"
            elif any(
                term in query_lower
                for term in ["ipc", "indian penal code", "भारतीय दण्ड संहिता"]
            ):
                requested_law = "IPC"
            elif any(
                term in query_lower
                for term in [
                    "bsa",
                    "bharatiya sakshya adhiniyam",
                    "bharatiya sakshya",
                    "sakshya",
                ]
            ):
                requested_law = "BSA"

            # Optional: Add more laws here later (e.g., IT Act, Consumer Protection, etc.)

            legal_keywords = [
                "section",
                "act",
                "law",
                "legal",
                "right",
                "remedy",
                "punishment",
                "penalty",
                "offense",
                "crime",
                "consumer",
                "protection",
                "domestic",
                "violence",
                "contract",
                "property",
                "cheating",
                "murder",
                "theft",
            ]

            enhanced_query = query
            if not any(keyword in query_lower for keyword in legal_keywords):
                enhanced_query += (
                    " legal law section act rights remedies punishment offense"
                )

            # Generate query embedding using the same function as the collection
            query_embedding = embedding_function([enhanced_query])[0]

            # Query the vector store
            results = self.vstore.query(query_embedding, n_results=k * 2)

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            print(
                f"Initial retrieval: {len(docs)} documents | Requested law: {requested_law}"
            )

            if not docs:
                return {
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                    "ids": [[]],
                }

            filtered_docs = []
            filtered_metas = []
            filtered_dists = []
            filtered_ids = []

            for i, (doc, meta, distance, chunk_id) in enumerate(
                zip(docs, metas, dists, ids)
            ):
                if doc and meta:
                    # PRIORITIZE REQUESTED LAW using metadata + content
                    if requested_law:
                        doc_act = meta.get("act", "").upper()
                        doc_text_lower = doc.lower()

                        law_matches = (
                            requested_law in doc_act
                            or requested_law.lower() in doc_text_lower
                            or requested_law
                            in meta.get("section", "")  # if section has BNS/IPC mention
                        )

                        if law_matches:
                            filtered_docs.append(doc)
                            filtered_metas.append(meta)
                            filtered_dists.append(distance)
                            filtered_ids.append(chunk_id)
                            continue  # skip further checks

                    # General relevance filtering
                    doc_lower = doc.lower()
                    has_query_terms = any(
                        term in doc_lower for term in query_lower.split()
                    )
                    has_legal_content = any(
                        keyword in doc_lower for keyword in legal_keywords
                    )
                    is_relevant_distance = distance < 1.0

                    if has_query_terms or (has_legal_content and is_relevant_distance):
                        filtered_docs.append(doc)
                        filtered_metas.append(meta)
                        filtered_dists.append(distance)
                        filtered_ids.append(chunk_id)

            # Fallback: take top k if filtering removed too many
            if not filtered_docs and docs:
                filtered_docs = docs[:k]
                filtered_metas = metas[:k]
                filtered_dists = dists[:k]
                filtered_ids = ids[:k]

            print(f"Final filtered: {len(filtered_docs)} documents")

            return {
                "documents": [filtered_docs[:k]],
                "metadatas": [filtered_metas[:k]],
                "distances": [filtered_dists[:k]],
                "ids": [filtered_ids[:k]],
            }

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]],
            }

    def generate_answer(
        self, question: str, retrieved: Dict
    ) -> Tuple[str, List[SourceItem]]:
        try:
            docs = retrieved.get("documents", [[]])[0]
            metas = retrieved.get("metadatas", [[]])[0]

            if not docs:
                context = "No relevant legal provisions found."
            else:
                context = self._format_context(docs, metas, [], [])

            # Get chat history from memory
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])

            # Format history as string
            history_text = ""
            for msg in chat_history[-6:]:  # Last 6 messages (3 turns)
                if msg.type == "human":
                    history_text += f"User: {msg.content}\n"
                elif msg.type == "ai":
                    history_text += f"Assistant: {msg.content}\n"

            # Final prompt with history
            formatted_prompt = self.qa_prompt.format(
                question=question,
                context=context,
                chat_history=history_text or "This is a new conversation.",
            )

            answer = self.provider.generate(formatted_prompt, max_tokens=1000)
            answer = self._clean_response(answer)

            return answer, []

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error: {str(e)}", []

    def query(
        self, question: str, top_k: int = 4, session_id: str = "default"
    ) -> ChatResponse:
        try:
            # Load history from memory
            memory_vars = self.memory.load_memory_variables({})
            session_memory = memory_vars.get("chat_history", [])

            # Enhance question with context from history
            enhanced_question = self._enhance_question_with_context(
                question, session_memory
            )

            retrieved = self.retrieve(enhanced_question, k=top_k)
            answer, sources = self.generate_answer(enhanced_question, retrieved)

            # Save to memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)

            # FULL transcript (for download) – keep separate
            self.conversation_transcripts.setdefault(session_id, []).append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "user_query": question,
                    "legal_response": answer,
                }
            )

            return ChatResponse(answer=answer, sources=sources)

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return ChatResponse(
                answer=f"Sorry, I encountered an error: {str(e)}",
                sources=[],
            )

    def get_full_transcript(self, session_id: str):
        return self.conversation_transcripts.get(session_id, [])

    def _enhance_question_with_context(self, question: str, history: list) -> str:
        """Enhance question with conversation context from LangChain memory"""
        if not history:
            return question

        question_lower = question.lower()

        # Check last 2-3 messages for context (efficient for short memory)
        for msg in reversed(history[-3:]):  # HumanMessage or AIMessage
            if hasattr(msg, "content"):  # Check if it's a message object
                prev_content_lower = msg.content.lower()

                # Look for section references
                import re

                section_matches = re.findall(
                    r"section\s+(\d+[a-z]*)", prev_content_lower
                )

                for section in section_matches:
                    if (
                        f"section {section}" in question_lower
                        or section in question_lower
                    ):
                        return f"""Previous context mentioned Section {section.upper()}. 
Current question: {question}
Note: If Section {section.upper()} was mentioned earlier but isn't in the legal documents, I might not have detailed information about it."""

                # Pronoun check
                if any(
                    pronoun in question_lower
                    for pronoun in ["this", "that", "it", "he", "she"]
                ):
                    return f"""Following up on previous response: {msg.content[:100]}...
Current question: {question}"""

        return question

    # Provides stats about Working
    def get_stats(self) -> Dict:
        try:
            total_chunks = self.vstore.collection.count()

            results = self.vstore.collection.get(include=["metadatas"])
            unique_docs = set()

            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata and "doc_id" in metadata:
                        unique_docs.add(metadata["doc_id"])

            total_documents = len(unique_docs)

            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "provider": type(self.provider).__name__,
                "collection": self.vstore.col_name,
                "embedding_model": getattr(self.provider, "model_name", "Unknown"),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        try:
            self.vstore.client.delete_collection(self.vstore.col_name)
            self.vstore.collection = self.vstore.client.create_collection(
                name=self.vstore.col_name, metadata={"source": "legal_docs"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_prompt_templates(self) -> Dict:
        return {
            "qa_prompt": self.qa_prompt.template[:100] + "...",
            "context_prompt": self.context_prompt.template[:100] + "...",
            "input_variables": {
                "qa_prompt": self.qa_prompt.input_variables,
                "context_prompt": self.context_prompt.input_variables,
            },
        }

    def health_check(self) -> Dict:
        try:
            doc_count = self.vstore.collection.count()

            provider_status = "healthy"
            try:
                test_embeddings = self.provider.get_embeddings(["test"])
                if not test_embeddings or len(test_embeddings) == 0:
                    provider_status = "unhealthy"
            except Exception as e:
                provider_status = f"unhealthy: {str(e)}"

            return {
                "status": "healthy",
                "vector_store_documents": doc_count,
                "provider_status": provider_status,
                "provider_name": type(self.provider).__name__,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def query_with_language(
        self,
        question: str,
        language: str = "en",
        top_k: int = 4,
        session_id: str = "default",
    ):
        try:
            logger.info(f"Processing question in {language}: {question}")

            processed_question = question
            if language != "en":
                try:
                    processed_question = translator.translate_legal_response(
                        question, "en"
                    )
                    logger.info(f"Translated question to English: {processed_question}")
                except Exception as trans_error:
                    logger.warning(
                        f"Question translation failed, using original: {trans_error}"
                    )

            response = self.query(
                question=processed_question, top_k=top_k, session_id=session_id
            )

            if language != "en":
                try:
                    translated_answer = translator.translate_legal_response(
                        response.answer, language
                    )

                    return ChatResponse(
                        answer=translated_answer, sources=response.sources
                    )

                except Exception as translation_error:
                    logger.error(f"Answer translation failed: {translation_error}")
                    return response
            else:
                return response

        except Exception as e:
            logger.error(f"Language query failed: {e}")
            error_msg = "Sorry, I encountered an error processing your question."
            if language != "en":
                try:
                    error_msg = translator.translate_legal_response(error_msg, language)
                except:
                    pass
            return ChatResponse(answer=error_msg, sources=[])


# Factory function to get configured RAG engine instance
def get_rag_engine(provider: ProviderClient = None) -> RAGEngine:
    return RAGEngine(provider)
