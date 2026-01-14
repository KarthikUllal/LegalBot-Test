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

from langchain.memory import ConversationBufferWindowMemory
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, provider: ProviderClient = None):
        self.provider = provider or get_best_provider()
        self.vstore = VectorStore()
        self.memories = {}  # session_id -> ConversationBufferWindowMemory
        self.conversation_transcripts = {}  # session_id -> transcript

        self._setup_prompts()

        logger.info(
            f"RAG Engine initialized with provider: {type(self.provider).__name__}"
        )

    def get_memory(self, session_id: str):
        """Get or create memory for this session"""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferWindowMemory(
                k=6, return_messages=True  # last 3 turns
            )
        return self.memories[session_id]

    def _setup_prompts(self):
        # Strong, detailed prompt like your original project
        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""You are "Nyaya Mitra" — a friendly, professional, and trusted AI legal assistant helping people understand Indian laws in simple, clear language.

Current User Message: "{question}"

Previous Conversation (use only if the current message clearly refers to it):
{chat_history}

Relevant Legal Provisions (your main source — use this fully for legal questions):
{context}

**Critical Instructions — Follow Exactly:**

1. Detect message type:
   - If the message is purely casual/greeting (hi, hello, namaste, hey, thank you, thanks, bye, good night, how are you, etc.) → 
     Reply warmly and briefly (1-2 sentences). Example: "Hello! How can I help you with Indian laws today?" or "You're welcome!"
     Do NOT give legal info or long explanation.

   - If the message is ANY legal question or situation (even short) → 
     Give a FULL, DETAILED, PRACTICAL answer using the legal provisions.
     Start directly with the answer — no long greeting.
     Use numbered steps for procedures.
     Mention exact section and act name.
     Be confident and complete.

2. For legal answers:
   - ALWAYS use all relevant context
   - Give clear numbered steps for any procedure (FIR, complaint, protection order)
   - For domestic violence: Explain Protection of Women from Domestic Violence Act, 2005 — Protection Officer, reliefs, orders
   - For consumer issues: Explain Consumer Protection Act, 2019 — rights, district commission
   - For theft/robbery/cheating: Use current BNS sections
   - BNS replaced IPC on 1 July 2024 — prefer BNS
   - NEVER say "I do not have sufficient details" if context exists — use it confidently
   - NEVER repeat sentences
   - Keep answer 4–10 sentences, structured and practical

3. General rules:
   - The user is always seeking help or is the victim — never assume they committed crime
   - Speak naturally like a trusted lawyer
   - Use simple language
   - Complete every answer — never stop mid-sentence
   - NEVER restate, repeat, or summarize the user's question at the beginning.
    - Start your answer directly with the advice or information.
    - Do NOT say things like "You asked about...", "The user has asked...", "Your question is...".
    - Jump straight into the helpful response.

Now respond appropriately based on the message type:""",
        )

    def _clean_response(self, response: str) -> str:
        response = re.sub(r"\n\s*\n", "\n\n", response)
        return response.strip()

    def _format_context(self, docs: List[str], metas: List[dict]) -> str:
        if not docs:
            return "No relevant legal provisions found."

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
            # Special label for Domestic Violence Act
            if "domestic violence" in act.lower():
                act = "Protection of Women from Domestic Violence Act, 2005"
            parts.append(f"From {act} ({section_text}):\n{doc.strip()}\n")

        return "\n\n".join(parts)

    def ingest_text(self, doc_id: str, text: str, metadata: dict = None) -> bool:
        try:
            logger.info(f"Ingesting document: {doc_id}")
            chunks = split_into_chunks(text)

            from .ingestion import enrich_metadata_with_section

            base_meta = {
                **(metadata or {}),
                "doc_id": doc_id,
                "act": metadata.get("act", doc_id) if metadata else doc_id,
            }
            metadatas = enrich_metadata_with_section(chunks, base_meta)

            ids = [f"{doc_id}__{i}" for i in range(len(chunks))]

            batch_size = 500
            for i in range(0, len(chunks), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_docs = chunks[i : i + batch_size]
                batch_metas = metadatas[i : i + batch_size]
                self.vstore.collection.add(
                    ids=batch_ids, documents=batch_docs, metadatas=batch_metas
                )

            logger.info(f"Successfully ingested {len(chunks)} chunks from {doc_id}")
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
            text = load_pdf_text(path)
            if not text:
                raise ValueError("No text extracted from PDF.")

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

            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            query_embedding = embedding_function([query])[0]

            results = self.vstore.query(query_embedding, n_results=k * 2)

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            if not docs:
                return {
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                    "ids": [[]],
                }

            return {
                "documents": [docs[:k]],
                "metadatas": [metas[:k]],
                "distances": [results.get("distances", [[]])[0][:k]],
                "ids": [results.get("ids", [[]])[0][:k]],
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
        self, question: str, retrieved: Dict, session_id: str
    ) -> Tuple[str, List[SourceItem]]:
        try:
            docs = retrieved.get("documents", [[]])[0]
            metas = retrieved.get("metadatas", [[]])[0]

            context = self._format_context(docs, metas)

            memory = self.get_memory(session_id)
            memory_vars = memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])

            history_text = (
                "\n".join(
                    [
                        f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
                        for msg in chat_history[-6:]
                    ]
                )
                or "This is a new conversation."
            )

            formatted_prompt = self.qa_prompt.format(
                question=question, context=context, chat_history=history_text
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
            memory = self.get_memory(session_id)

            retrieved = self.retrieve(question, k=top_k)
            answer, sources = self.generate_answer(question, retrieved, session_id)

            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)

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
            return ChatResponse(answer="Sorry, error occurred.", sources=[])

    def query_with_language(
        self,
        question: str,
        language: str = "en",
        top_k: int = 4,
        session_id: str = "default",
    ) -> ChatResponse:
        try:
            # Use original question for retrieval — no translation
            response = self.query(question=question, top_k=top_k, session_id=session_id)

            if language != "en":
                try:
                    response.answer = translator.translate_legal_response(
                        response.answer, language
                    )
                except:
                    pass  # Keep English if translation fails

            return response
        except Exception as e:
            logger.error(f"Language query failed: {e}")
            return ChatResponse(answer="Error processing request.", sources=[])

    def get_full_transcript(self, session_id: str):
        return self.conversation_transcripts.get(session_id, [])

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
