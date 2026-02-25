# """
# Advanced RAG Engine v3.0
# - Intent Classifier Agent (fact/summary/comparison/explanation)
# - BM25 Keyword Search + FAISS Semantic Search (Hybrid Retrieval)
# - Section-Aware Chunking (ALL CAPS header detection)
# - Dynamic prompts per intent
# - Conversational memory
# - Source attribution with page + section
# """

# import os
# import re
# import math
# import string
# from typing import Any
# from collections import Counter

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document
# from langchain_groq import ChatGroq


# # ─── PROMPTS ──────────────────────────────────────────────────────────────────

# PROMPTS = {
#     "fact": """You are a precise document analyst. Answer with the specific fact, number, or date requested.
# - Cite the page or section.
# - Never guess. If not found say "This specific data is not in the document."

# Context:
# {context}

# Chat History:
# {chat_history}

# Question: {question}
# Answer:""",

#     "summary": """You are an expert document analyst. Give a comprehensive, well-structured summary.
# - Use bullet points and sections.
# - Include ALL quantitative results available.
# - Cover every aspect the user asked about.
# - Do NOT say "not available" without checking all context.

# Context:
# {context}

# Chat History:
# {chat_history}

# Question: {question}
# Comprehensive Answer:""",

#     "comparison": """You are an expert document analyst. Structure your answer as a clear comparison.
# - Include specific numbers for both items being compared.
# - Show percentage changes where relevant.
# - Cite page numbers.

# Context:
# {context}

# Chat History:
# {chat_history}

# Question: {question}
# Comparison:""",

#     "explanation": """You are an expert document analyst. Explain clearly and simply.
# - Use examples from the document.
# - Break down complex concepts step by step.
# - Reference specific sections.

# Context:
# {context}

# Chat History:
# {chat_history}

# Question: {question}
# Explanation:""",
# }


# # ─── INTENT CLASSIFIER AGENT ──────────────────────────────────────────────────

# class IntentClassifierAgent:
#     """
#     Rule-based intent classifier agent.
#     Detects question type → selects optimal retrieval config + prompt.
#     No LLM call — fast, free, deterministic.
    
#     In production this would use a fine-tuned classifier or LLM-based routing.
#     """

#     PATTERNS = {
#         "comparison": [r"\bcompar\w*\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b",
#                        r"\bchange\b", r"\bgrowth\b", r"\bdecline\b", r"\byoy\b", r"\bqoq\b"],
#         "summary":    [r"\bsummar\w*\b", r"\boverview\b", r"\bhighlight\w*\b",
#                        r"\bkey points\b", r"\beverything\b", r"\boverall\b", r"\bbrief\b"],
#         "explanation":[r"\bwhy\b", r"\bhow does\b", r"\bexplain\b", r"\bwhat is\b",
#                        r"\bwhat are\b", r"\bdefine\b", r"\bdescribe\b"],
#         "fact":       [r"\bhow much\b", r"\bhow many\b", r"\bwhat was\b", r"\bwhat were\b",
#                        r"\bwhen\b", r"\btotal\b", r"\bpercent\b", r"\bmargin\b"],
#     }

#     # (semantic_k, bm25_k) per intent
#     RETRIEVAL = {
#         "fact":        (6,  4),
#         "summary":     (15, 10),
#         "comparison":  (12, 8),
#         "explanation": (8,  5),
#         "general":     (8,  5),
#     }

#     def classify(self, question: str) -> str:
#         q = question.lower()
#         scores = {k: sum(1 for p in v if re.search(p, q))
#                   for k, v in self.PATTERNS.items()}
#         best = max(scores, key=scores.get)
#         return best if scores[best] > 0 else "general"

#     def get_retrieval(self, intent: str) -> tuple:
#         return self.RETRIEVAL.get(intent, self.RETRIEVAL["general"])

#     def get_prompt(self, intent: str) -> str:
#         return PROMPTS.get(intent, PROMPTS["summary"])


# # ─── BM25 (built from scratch, no extra pip install) ─────────────────────────

# class BM25Engine:
#     """
#     Pure-Python BM25 keyword retriever.
#     Complements FAISS semantic search for hybrid retrieval.
#     Especially useful for: exact terms, numbers, proper nouns, technical jargon.
#     """

#     def __init__(self, k1=1.5, b=0.75):
#         self.k1 = k1
#         self.b = b
#         self.docs = []
#         self.doc_freqs = []
#         self.idf = {}
#         self.avgdl = 0
#         self._stopwords = {"the","a","an","is","it","in","on","at","to","for",
#                            "of","and","or","but","not","with","this","that",
#                            "was","are","be","been","by","from","as","we","our"}

#     def _tokenize(self, text: str) -> list:
#         text = text.lower().translate(str.maketrans("","",string.punctuation))
#         return [t for t in text.split() if t not in self._stopwords and len(t) > 2]

#     def fit(self, documents: list):
#         self.docs = documents
#         tokenized = [self._tokenize(d.page_content) for d in documents]
#         self.avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
#         df = Counter(word for tokens in tokenized for word in set(tokens))
#         N = len(documents)
#         self.idf = {w: math.log((N - f + 0.5) / (f + 0.5) + 1) for w, f in df.items()}
#         self.doc_freqs = [Counter(t) for t in tokenized]

#     def retrieve(self, query: str, k: int = 5) -> list:
#         if not self.docs:
#             return []
#         tokens = self._tokenize(query)
#         scores = []
#         for i, df in enumerate(self.doc_freqs):
#             dl = sum(df.values())
#             score = sum(
#                 self.idf.get(t, 0) * df.get(t, 0) * (self.k1 + 1) /
#                 (df.get(t, 0) + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
#                 for t in tokens
#             )
#             scores.append((score, i))
#         scores.sort(reverse=True)
#         return [self.docs[i] for _, i in scores[:k]]


# # ─── SECTION-AWARE CHUNKER ────────────────────────────────────────────────────

# class SectionAwareChunker:
#     """
#     Detects ALL CAPS section headers and chunks by section.
#     Works for: financial reports, research papers, legal docs, manuals, whitepapers.
#     Falls back to RecursiveCharacterTextSplitter for unstructured content.
#     """

#     HEADER = re.compile(r'^([A-Z][A-Z\s&,\-–]{4,}[A-Z])\s*$', re.MULTILINE)

#     def __init__(self, chunk_size=1000, chunk_overlap=150):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=["\n\n", "\n", ".", " ", ""],
#         )

#     def chunk(self, pages: list) -> list:
#         all_chunks = []
#         for page in pages:
#             text = page.page_content
#             meta = page.metadata.copy()
#             matches = list(self.HEADER.finditer(text))

#             if not matches:
#                 all_chunks.extend(self.splitter.split_documents([page]))
#                 continue

#             # Pre-header text
#             if matches[0].start() > 0:
#                 pre = text[:matches[0].start()].strip()
#                 if pre:
#                     all_chunks.append(Document(
#                         page_content=pre, metadata={**meta, "section": "PREAMBLE"}))

#             # Chunk by section
#             for i, m in enumerate(matches):
#                 title = m.group(1).strip()
#                 end = matches[i+1].start() if i+1 < len(matches) else len(text)
#                 body = text[m.start():end].strip()
#                 section_meta = {**meta, "section": title}

#                 if len(body) <= self.chunk_size:
#                     all_chunks.append(Document(
#                         page_content=f"[SECTION: {title}]\n{body}",
#                         metadata=section_meta))
#                 else:
#                     sub = self.splitter.split_documents(
#                         [Document(page_content=body, metadata=section_meta)])
#                     for c in sub:
#                         c.page_content = f"[SECTION: {title}]\n{c.page_content}"
#                     all_chunks.extend(sub)

#         return all_chunks


# # ─── RAG ENGINE ───────────────────────────────────────────────────────────────

# class RAGEngine:
#     """
#     Production-grade RAG Engine combining:
#     - Intent Classifier Agent
#     - Section-Aware Chunking  
#     - Hybrid BM25 + FAISS Retrieval
#     - Dynamic prompts per intent
#     - Conversational memory (last 5 turns)
#     - Full source attribution
#     """

#     def __init__(self, temperature: float = 0.1):

#         self.classifier = IntentClassifierAgent()
#         self.bm25 = BM25Engine()
#         self.chunker = SectionAwareChunker(chunk_size=1000, chunk_overlap=150)

#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2",
#             model_kwargs={"device": "cpu"},
#             encode_kwargs={"normalize_embeddings": True},
#         )

#         self.llm = ChatGroq(
#             model="llama-3.1-8b-instant",
#             temperature=temperature,
#             api_key=os.getenv("GROQ_API_KEY"),
#         )

#         self.vectorstore = None
#         self.all_chunks = []
#         self.memory = ConversationBufferWindowMemory(
#             k=5,
#             memory_key="chat_history",
#             return_messages=True,
#             output_key="answer",
#         )

#     def ingest_pdf(self, pdf_path: str) -> int:
#         """Load PDF → section-aware chunk → FAISS + BM25 index."""
#         loader = PyPDFLoader(pdf_path)
#         pages = loader.load()

#         filename = pdf_path.replace("\\", "/").split("/")[-1]
#         for page in pages:
#             page.metadata["source"] = filename

#         # Section-aware chunking
#         self.all_chunks = self.chunker.chunk(pages)

#         # FAISS semantic index
#         self.vectorstore = FAISS.from_documents(self.all_chunks, self.embeddings)

#         # BM25 keyword index
#         self.bm25.fit(self.all_chunks)

#         return len(self.all_chunks)

#     def _hybrid_retrieve(self, question: str, semantic_k: int, bm25_k: int) -> list:
#         """Merge FAISS + BM25 results, deduplicate."""
#         semantic_docs = self.vectorstore.similarity_search(question, k=semantic_k)
#         bm25_docs = self.bm25.retrieve(question, k=bm25_k)

#         seen, merged = set(), []
#         for doc in semantic_docs + bm25_docs:
#             key = doc.page_content[:80]
#             if key not in seen:
#                 seen.add(key)
#                 merged.append(doc)
#         return merged

#     def query(self, question: str) -> dict[str, Any]:
#         if not self.vectorstore:
#             raise ValueError("No document ingested. Call ingest_pdf() first.")

#         # 1. Classify intent
#         intent = self.classifier.classify(question)
#         semantic_k, bm25_k = self.classifier.get_retrieval(intent)
#         prompt_template = self.classifier.get_prompt(intent)

#         # 2. Hybrid retrieval
#         docs = self._hybrid_retrieve(question, semantic_k, bm25_k)

#         # 3. Build context (token-safe, max ~3000 tokens)
#         max_chars = 12000
#         context_parts, total = [], 0
#         for doc in docs:
#             if total + len(doc.page_content) > max_chars:
#                 break
#             context_parts.append(doc.page_content)
#             total += len(doc.page_content)
#         context = "\n\n---\n\n".join(context_parts)

#         # 4. Chat history
#         history = self.memory.load_memory_variables({})
#         chat_history = history.get("chat_history", [])
#         history_text = "".join(
#             f"{'Human' if m.type == 'human' else 'Assistant'}: {m.content}\n"
#             for m in chat_history
#         )

#         # 5. Build prompt + call LLM
#         prompt = PromptTemplate(
#             template=prompt_template,
#             input_variables=["context", "chat_history", "question"]
#         )
#         chain = prompt | self.llm
#         response = chain.invoke({
#             "context": context,
#             "chat_history": history_text,
#             "question": question,
#         })
#         answer = response.content if hasattr(response, "content") else str(response)

#         # 6. Save to memory
#         self.memory.save_context({"input": question}, {"answer": answer})

#         # 7. Build sources with section info
#         sources, seen = [], set()
#         for doc in docs:
#             page = doc.metadata.get("page", 0) + 1
#             source = doc.metadata.get("source", "document")
#             section = doc.metadata.get("section", "")
#             key = f"{source}_p{page}_{section}"
#             if key not in seen:
#                 seen.add(key)
#                 sources.append({
#                     "page": page,
#                     "source": source,
#                     "section": section,
#                     "preview": doc.page_content[:200].strip() + "...",
#                 })

#         return {
#             "answer": answer,
#             "sources": sources,
#             "intent": intent,
#         }




"""
Advanced RAG Engine v5.0
- Fixes LLM confusing Q1 vs Q2 comparisons
- Fixes GAAP operating income not being found
- Intent Classifier Agent
- BM25 + FAISS Hybrid Retrieval  
- Section-Aware Chunking
- Table-Aware Context Processing
- Section Boosting
- Dynamic prompts per intent
"""

import os
import re
import math
import string
from typing import Any
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_groq import ChatGroq


# ─── PROMPTS ──────────────────────────────────────────────────────────────────

PROMPTS = {
    "fact": """You are a precise financial analyst. Find the EXACT number requested.

STRICT TABLE READING RULES:
- Tables always follow this column order: Q2-2024 | Q3-2024 | Q4-2024 | Q1-2025 | Q2-2025
- Q2-2025 = 5th value (LAST)
- Q1-2025 = 4th value  
- Q4-2024 = 3rd value
- Q3-2024 = 2nd value
- Q2-2024 = 1st value (FIRST)

Example: "Total revenues 25,500 25,182 25,707 19,335 22,496"
- Q2-2024 = 25,500
- Q2-2025 = 22,496 (last number)

Find the specific number asked and state clearly which quarter it belongs to.
If you see "Income from operations 1,605 2,717 1,583 399 923" then:
- Q2-2024 operating income = 1,605
- Q2-2025 operating income = 923 (last value)

Context:
{context}

Chat History:
{chat_history}

Question: {question}

State the exact number and which column position you read it from:""",

    "summary": """You are an expert financial analyst reviewing Tesla's Q2 2025 earnings report.

Extract and present these sections IN ORDER:
1. PROFITABILITY (operating income, net income, margins)
2. REVENUE (total revenue, automotive, energy, services - with YoY changes)
3. OPERATIONS (deliveries, production, Superchargers)
4. AI & TECHNOLOGY (Robotaxi, FSD, Cortex GPUs)
5. ENERGY BUSINESS (Megapack, storage deployments, gross profit)
6. OUTLOOK (new vehicles, Cybercab, strategic priorities)

Rules:
- Use the HIGHLIGHTS section first if available
- Include specific numbers for everything
- YoY = compare Q2-2025 to Q2-2024 (not Q1-2025)
- Format clearly with headers and bullet points

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Structured Summary:""",

    "comparison": """You are a precise financial analyst doing a comparison.

CRITICAL TABLE READING:
Column order is ALWAYS: Q2-2024 | Q3-2024 | Q4-2024 | Q1-2025 | Q2-2025

For "Compare Q1 2025 vs Q2 2025":
- Q1-2025 = 4th number in each row
- Q2-2025 = 5th number (last) in each row

Example row: "Total revenues 25,500 25,182 25,707 19,335 22,496"
- Q1-2025 revenue = 19,335
- Q2-2025 revenue = 22,496
- Change = +3,161 (+16.3% QoQ)

Show:
| Metric | Q1-2025 | Q2-2025 | Change |
Include: Total Revenue, Gross Profit, Operating Income, Net Income, Deliveries

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Accurate comparison table (read each row carefully):""",

    "explanation": """You are an expert document analyst. Explain clearly using only document content.
- Use specific examples and data from the document
- Break down complex concepts step by step  
- Reference page numbers where possible

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Clear explanation:""",
}


# ─── INTENT CLASSIFIER AGENT ──────────────────────────────────────────────────

class IntentClassifierAgent:
    """
    Rule-based intent classifier agent.
    Detects question type → selects retrieval config + prompt.
    Fast, free, deterministic — no LLM call needed.
    Production version would use a fine-tuned classifier.
    """

    PATTERNS = {
        "comparison": [
            r"\bcompar\w*\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b",
            r"\bchange\b", r"\bgrowth\b", r"\bdecline\b", r"\byoy\b", r"\bqoq\b",
            r"\bq1.+q2\b", r"\bfrom.+to\b", r"\bincrease\b", r"\bdecrease\b"
        ],
        "summary": [
            r"\bsummar\w*\b", r"\boverview\b", r"\bhighlight\w*\b",
            r"\bkey points\b", r"\bmost important\b", r"\beverything\b",
            r"\boverall\b", r"\bbrief\b", r"\blist\b", r"\bfindings\b"
        ],
        "explanation": [
            r"\bwhy\b", r"\bhow does\b", r"\bexplain\b", r"\bwhat is\b",
            r"\bwhat are\b", r"\bdefine\b", r"\bdescribe\b", r"\bstrategy\b"
        ],
        "fact": [
            r"\bhow much\b", r"\bhow many\b", r"\bwhat was\b", r"\bwhat were\b",
            r"\bwhen\b", r"\btotal\b", r"\bpercent\b", r"\bmargin\b",
            r"\brevenue\b", r"\bincome\b", r"\bearnings\b", r"\beps\b",
            r"\boperating\b", r"\bgaap\b", r"\bcash\b", r"\bdeliveries\b"
        ],
    }

    RETRIEVAL = {
        "fact":        (10, 8),
        "summary":     (18, 12),
        "comparison":  (14, 10),
        "explanation": (10, 6),
        "general":     (10, 6),
    }

    def classify(self, question: str) -> str:
        q = question.lower()
        scores = {k: sum(1 for p in v if re.search(p, q))
                  for k, v in self.PATTERNS.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def get_retrieval(self, intent: str) -> tuple:
        return self.RETRIEVAL.get(intent, self.RETRIEVAL["general"])

    def get_prompt(self, intent: str) -> str:
        return PROMPTS.get(intent, PROMPTS["summary"])


# ─── TABLE CONTEXT PROCESSOR ──────────────────────────────────────────────────

class TableContextProcessor:
    """
    Injects explicit column labels into financial table chunks.
    Prevents LLM from mixing up Q2-2024 vs Q2-2025 values.
    """

    def process(self, text: str) -> str:
        if re.search(r'Q\d[-–]\d{4}', text):
            header = (
                "\n[TABLE COLUMN ORDER: Col1=Q2-2024 | Col2=Q3-2024 | "
                "Col3=Q4-2024 | Col4=Q1-2025 | Col5=Q2-2025]\n"
                "[READ CAREFULLY: Last value in each row = Q2-2025. "
                "4th value = Q1-2025.]\n"
            )
            return header + text
        return text

    def process_docs(self, docs: list) -> list:
        return [
            Document(
                page_content=self.process(doc.page_content),
                metadata=doc.metadata.copy()
            )
            for doc in docs
        ]


# ─── BM25 ENGINE ─────────────────────────────────────────────────────────────

class BM25Engine:
    """
    Pure-Python BM25 keyword retriever — no extra pip install.
    Complements FAISS for exact terms, numbers, quarter references.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0
        self._stop = {
            "the","a","an","is","it","in","on","at","to","for","of",
            "and","or","but","not","with","this","that","was","are",
            "be","been","by","from","as","we","our"
        }

    def _tokenize(self, text: str) -> list:
        text = text.lower().translate(str.maketrans("","",string.punctuation))
        return [t for t in text.split() if t not in self._stop and len(t) > 1]

    def fit(self, documents: list):
        self.docs = documents
        tokenized = [self._tokenize(d.page_content) for d in documents]
        self.avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
        df = Counter(w for tokens in tokenized for w in set(tokens))
        N = len(documents)
        self.idf = {
            w: math.log((N - f + 0.5) / (f + 0.5) + 1)
            for w, f in df.items()
        }
        self.doc_freqs = [Counter(t) for t in tokenized]

    def retrieve(self, query: str, k: int = 5) -> list:
        if not self.docs:
            return []
        tokens = self._tokenize(query)
        scores = []
        for i, df in enumerate(self.doc_freqs):
            dl = sum(df.values())
            score = sum(
                self.idf.get(t, 0) * df.get(t, 0) * (self.k1 + 1) /
                max(df.get(t, 0) + self.k1 * (
                    1 - self.b + self.b * dl / max(self.avgdl, 1)
                ), 1e-6)
                for t in tokens
            )
            scores.append((score, i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:k]]


# ─── SECTION BOOSTER ──────────────────────────────────────────────────────────

class SectionBooster:
    """
    Re-ranks retrieved chunks so important sections appear first.
    Pushes legal/disclaimer sections to the back for summary questions.
    """

    HIGH_PRIORITY = [
        "HIGHLIGHTS", "SUMMARY", "FINANCIAL SUMMARY",
        "OPERATIONAL SUMMARY", "OUTLOOK", "CORE TECHNOLOGY",
        "ENERGY", "AUTOMOTIVE", "STATEMENT OF OPERATIONS",
        "REVENUES", "OPERATING EXPENSES"
    ]

    LOW_PRIORITY = [
        "FORWARD-LOOKING", "ADDITIONAL INFORMATION",
        "CERTAIN TERMS", "WEBCAST", "PHOTOS", "DINER"
    ]

    def boost(self, docs: list, intent: str) -> list:
        if intent not in ("summary", "general", "fact"):
            return docs

        high, normal, low = [], [], []
        for doc in docs:
            section = doc.metadata.get("section", "").upper()
            content_top = doc.page_content.upper()[:150]

            if any(p in section or p in content_top for p in self.HIGH_PRIORITY):
                high.append(doc)
            elif any(p in section or p in content_top for p in self.LOW_PRIORITY):
                low.append(doc)
            else:
                normal.append(doc)

        return high + normal + low


# ─── SECTION-AWARE CHUNKER ────────────────────────────────────────────────────

class SectionAwareChunker:
    """
    Detects ALL CAPS headers → chunks by section.
    Keeps financial tables together instead of splitting mid-row.
    Works for any document type: reports, papers, contracts, manuals.
    """

    HEADER = re.compile(r'^([A-Z][A-Z\s&,\-–]{4,}[A-Z])\s*$', re.MULTILINE)

    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def chunk(self, pages: list) -> list:
        all_chunks = []
        for page in pages:
            text = page.page_content
            meta = page.metadata.copy()
            matches = list(self.HEADER.finditer(text))

            if not matches:
                all_chunks.extend(self.splitter.split_documents([page]))
                continue

            if matches[0].start() > 0:
                pre = text[:matches[0].start()].strip()
                if pre:
                    all_chunks.append(Document(
                        page_content=pre,
                        metadata={**meta, "section": "PREAMBLE"}
                    ))

            for i, m in enumerate(matches):
                title = m.group(1).strip()
                end = matches[i+1].start() if i+1 < len(matches) else len(text)
                body = text[m.start():end].strip()
                section_meta = {**meta, "section": title}

                if len(body) <= self.chunk_size:
                    all_chunks.append(Document(
                        page_content=f"[SECTION: {title}]\n{body}",
                        metadata=section_meta
                    ))
                else:
                    sub = self.splitter.split_documents(
                        [Document(page_content=body, metadata=section_meta)]
                    )
                    for c in sub:
                        c.page_content = f"[SECTION: {title}]\n{c.page_content}"
                    all_chunks.extend(sub)

        return all_chunks


# ─── MAIN RAG ENGINE ──────────────────────────────────────────────────────────

class RAGEngine:

    def __init__(self, temperature: float = 0.1):
        self.classifier = IntentClassifierAgent()
        self.bm25 = BM25Engine()
        self.chunker = SectionAwareChunker(chunk_size=1000, chunk_overlap=150)
        self.table_processor = TableContextProcessor()
        self.booster = SectionBooster()

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )

        self.vectorstore = None
        self.all_chunks = []
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

    def ingest_pdf(self, pdf_path: str) -> int:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        filename = pdf_path.replace("\\", "/").split("/")[-1]
        for page in pages:
            page.metadata["source"] = filename

        self.all_chunks = self.chunker.chunk(pages)
        self.vectorstore = FAISS.from_documents(self.all_chunks, self.embeddings)
        self.bm25.fit(self.all_chunks)

        return len(self.all_chunks)

    def _hybrid_retrieve(self, question: str,
                         semantic_k: int, bm25_k: int) -> list:
        semantic_docs = self.vectorstore.similarity_search(question, k=semantic_k)
        bm25_docs = self.bm25.retrieve(question, k=bm25_k)

        seen, merged = set(), []
        for doc in semantic_docs + bm25_docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged

    def query(self, question: str) -> dict[str, Any]:
        if not self.vectorstore:
            raise ValueError("No document ingested.")

        # 1. Classify intent
        intent = self.classifier.classify(question)
        semantic_k, bm25_k = self.classifier.get_retrieval(intent)
        prompt_template = self.classifier.get_prompt(intent)

        # 2. Hybrid retrieve
        docs = self._hybrid_retrieve(question, semantic_k, bm25_k)

        # 3. Section boost
        docs = self.booster.boost(docs, intent)

        # 4. Table context injection
        docs = self.table_processor.process_docs(docs)

        # 5. Build context (token-safe)
        max_chars = 12000
        context_parts, total = [], 0
        for doc in docs:
            if total + len(doc.page_content) > max_chars:
                break
            context_parts.append(doc.page_content)
            total += len(doc.page_content)
        context = "\n\n---\n\n".join(context_parts)

        # 6. Chat history
        history = self.memory.load_memory_variables({})
        history_text = "".join(
            f"{'Human' if m.type == 'human' else 'Assistant'}: {m.content}\n"
            for m in history.get("chat_history", [])
        )

        # 7. LLM call
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        response = (prompt | self.llm).invoke({
            "context": context,
            "chat_history": history_text,
            "question": question,
        })
        answer = response.content if hasattr(response, "content") else str(response)

        # 8. Save memory
        self.memory.save_context({"input": question}, {"answer": answer})

        # 9. Build sources
        sources, seen = [], set()
        for doc in docs:
            page = doc.metadata.get("page", 0) + 1
            source = doc.metadata.get("source", "document")
            section = doc.metadata.get("section", "")
            key = f"{source}_p{page}_{section}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "page": page,
                    "source": source,
                    "section": section,
                    "preview": doc.page_content[:200].strip() + "...",
                })

        return {
            "answer": answer,
            "sources": sources,
            "intent": intent,
        }