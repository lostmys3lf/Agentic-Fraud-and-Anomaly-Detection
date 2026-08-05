"""
L2 Investigation -- retrieval over the SOP corpus in fraud_policy_docs/.

Answers "what does policy say about this case?". Its output is what makes an L3 decision
citable: a risk level with no SOP clause behind it is an opinion, not a policy decision.

Same working order as L1: prototype the steps in notebooks/03_agent_pipeline_end_to_end.ipynb
first, then move the working logic in here. The signatures below are the target contract,
not a suggestion of where to start.

The four steps this module owns:
  1. load_sop_documents()  -- read the 4 SOP .txt files, split header from body.
  2. chunk_documents()     -- one chunk per numbered SOP section, each carrying enough
                              metadata to be cited as "SOP-001 §2 (Risk Indicators)".
  3. build_sop_index()     -- embed the chunks into a local Chroma collection.
  4. retrieve_vector()     -- semantic search, measured against
     retrieve_keyword()    -- a TF-IDF baseline.

Why two retrievers: the corpus is 4 short documents. An embedding index may well not beat
plain keyword matching at that size. CLAUDE.md requires reporting that comparison honestly
either way -- the vector store is here to be measured, not assumed.

Environment note (already paid for, do not re-litigate): chromadb is pinned to 0.6.3 and
onnxruntime to 1.20.1 in requirements.txt. chromadb 1.x segfaults on Windows (0xC0000005,
no traceback) once pandas or scikit-learn has loaded its bundled msvcp140.dll, because the
first copy loaded wins for the whole process and 1.x's Rust core needs a newer one.
Neither problem exists on Colab/Linux.
"""

from dataclasses import dataclass

import chromadb

import config
import os
import re
from chromadb.utils import embedding_functions

@dataclass(frozen=True)
class SopDocument:
    """One raw policy file, header already parsed off the body."""

    doc_id: str        # "SOP-001"
    doc_title: str     # "SIM Swap Verification and Escalation Procedure"
    body: str          # everything after the "Title:" line
    path: str


@dataclass(frozen=True)
class SopChunk:
    """
    One retrievable unit of policy text, normally one numbered SOP section.

    This is the data contract for the whole layer, so keep the fields stable: L3 compares
    against the clause, L4 prints the citation.
    """

    chunk_id: str        # stable id, so re-indexing overwrites instead of duplicating
    doc_id: str
    doc_title: str
    section_no: int
    section_title: str
    text: str

    @property
    def citation(self) -> str:
        """
        TODO: return the string an L3 decision or L4 report quotes as its source,
        in the form "SOP-001 §2 (Risk Indicators)". One document in the corpus has no
        numbered sections -- decide what it should look like and handle it here.

        Hint: f-string
        """
        if self.section_no == 0:
            return f"{self.doc_id} ({self.section_title})"
        return f"{self.doc_id} - {self.section_no} ({self.section_title})"
    
    

@dataclass(frozen=True)
class RetrievedChunk:
    """A hit: the chunk, how similar it was, and where it ranked."""

    chunk: SopChunk
    score: float   # similarity, higher is closer
    rank: int      # 1 = best hit for this query

class SopEmbedder:
    """Pembungkus embedding buat korpus SOP (L2).

    Ganti model = install library-nya, terus ganti satu baris self.backend di
    __init__. Selain baris itu nggak ada yang berubah: Step 9-11 tetap sama,
    karena semua backend dipanggil lewat kontrak yang sama (teks masuk, vektor
    keluar).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

        # ===== SATU-SATUNYA BARIS YANG DIGANTI KALAU TUKAR MODEL =====
        self.backend = embedding_functions.DefaultEmbeddingFunction()

        # pengganti, kalau nanti mau pakai transformer:
        #   pip install sentence-transformers
        #   self.backend = embedding_functions.SentenceTransformerEmbeddingFunction(
        #       model_name=model_name)
        # habis ganti: hapus folder config.CHROMA_PERSIST_DIR, jalanin ulang Step 9-10.
        # Vektor model lama dan model baru nggak bisa dicampur.
        # =============================================================

    def __call__(self, input: list[str]) -> list[list[float]]:
        hasil = self.backend(input)

        clean = []
        for h in hasil:
            clean.append(h.tolist())     # numpy array -> list float biasa

        return clean

def load_sop_documents(policy_dir: str = config.POLICY_DOCS_DIR) -> list[SopDocument]:
    sop_docs = []
    for name in sorted(os.listdir(policy_dir)):
        if not name.endswith('.txt'):
            continue
        path = os.path.join(policy_dir, name)
        with open(path, encoding='utf-8') as f:
            text = f.read()
        lines = text.splitlines() 
        
        m = re.search(r'SOP-\d{3}', lines[0]) #kenapa re.search dan bukan re.match (karena baris pertama SOP-004 formatnya beda sendiri).
        doc_id = m.group(0)

        doc_title = lines[1].removeprefix('Title: ') 
        body = '\n'.join(lines[2:]).strip()

        sop_docs.append(SopDocument(
            doc_id=doc_id,
            doc_title=doc_title,
            body=body,
            path=path
        )) #ini gunanya buat nyimpen data dokumen SOP ke dalam list sop_docs dalam be


    return sop_docs


def chunk_documents(documents: list[SopDocument]) -> list[SopChunk]:
    all_chunks = []
    for doc in documents:
        chunks = []
        current = None

        for line in doc.body.splitlines():
            m = re.match(r'^(\d+)\.\s+(.+)$', line)
            if m:
                if current is not None:
                    chunks.append(current)
                current = {
                    'section_no': int(m.group(1)),
                    'section_title': m.group(2).strip(),
                    'lines': [],
                }
            else:
                if current is not None:
                    current['lines'].append(line)

        if current is not None:
            chunks.append(current)

        # SOP-004 nggak punya bagian bernomor -> jadiin satu potongan utuh
        if not chunks:
            chunks = [{
                'section_no': 0,
                'section_title': 'Full Template',
                'lines': doc.body.splitlines(),
            }]

        for c in chunks:
            isi = '\n'.join(c['lines']).strip()
            all_chunks.append(SopChunk(
            chunk_id=f"{doc.doc_id}_{c['section_no']}",              # gabungan doc.doc_id dan c['section_no']
            doc_id=doc.doc_id,
            doc_title=doc.doc_title,
            section_no=c['section_no'],
            section_title=c['section_title'],
            text=isi
            )
        )
            # stable id, so re-indexing overwrites instead of duplicatin
    return all_chunks

   
def get_embedding_function(model_name: str = config.EMBEDDING_MODEL_NAME):
    return SopEmbedder(model_name=model_name)


def get_sop_collection(
    persist_dir: str = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.CHROMA_COLLECTION_SOP,
):
    """
    TODO: open (or create) the persisted Chroma collection, so a later run can query an
    index built earlier instead of re-embedding everything.

    Ask for cosine distance instead of Chroma's default L2: MiniLM embeddings carry
    meaning in direction, and the SOP sections vary a lot in length -- which L2 distance
    punishes and cosine ignores.

    Hint: `chromadb.PersistentClient(path=...)`,
          `get_or_create_collection(name=..., embedding_function=..., metadata={'hnsw:space': ...})`
    """
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={'hnsw:space': 'cosine'}
    )


def build_sop_index(
    chunks: list[SopChunk],
    persist_dir: str = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.CHROMA_COLLECTION_SOP,
    rebuild: bool = False,
):
    """
    TODO: embed the chunks, write them to the collection, return the collection.

      1. If rebuild is True, drop the existing collection first (needed after editing an
         SOP file, so chunks that no longer exist stop being retrievable).
      2. Write ids, documents and metadatas. Chroma only accepts str/int/float/bool as
         metadata values -- no nested objects, so flatten the chunk yourself.
      3. Prefer the write that is safe to re-run: a notebook cell gets executed more than
         once, and duplicated chunks would all match the same query.

    Hint: `client.delete_collection()`, `collection.upsert(ids=..., documents=..., metadatas=...)`
    """
    client = chromadb.PersistentClient(path=persist_dir)
    if rebuild:
        try:
            client.delete_collection(name=collection_name)
        except ValueError:
            pass
    collection = get_sop_collection(persist_dir=persist_dir, collection_name=collection_name)
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            'doc_id': c.doc_id,
            'doc_title': c.doc_title,
            'section_no': c.section_no,
            'section_title': c.section_title,
        }
        for c in chunks
    ]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return collection   
    


def retrieve_vector(
    query: str,
    collection,
    top_k: int = config.RAG_TOP_K,
    doc_id: str | None = None,
) -> list[RetrievedChunk]:
    """
    TODO: semantic search over the SOP index.

      1. Query the collection for top_k hits.
      2. Optionally restrict to one document first -- useful once L3 already knows the
         fraud category and only wants that SOP's clauses.
      3. Chroma returns cosine *distance*; convert it so higher means closer, and rebuild
         a SopChunk from the stored metadata + document.

    Careful when reporting: this score and retrieve_keyword()'s score come from different
    vector spaces and are not comparable in magnitude. Compare rankings, not numbers.

    Hint: `collection.query(query_texts=[...], n_results=..., where={...})`, `enumerate(..., start=1)`
    """
    raise NotImplementedError


def retrieve_keyword(
    query: str,
    chunks: list[SopChunk],
    top_k: int = config.RAG_TOP_K,
) -> list[RetrievedChunk]:
    """
    TODO: TF-IDF baseline -- the number the vector index has to beat (baseline first).

      1. Fit a TF-IDF vectorizer over the chunk texts, transform the query.
      2. Score every chunk by cosine similarity, return the top_k as RetrievedChunk.

    Refitting on every call is wasteful in general but free here (~13 short chunks), and
    it keeps the function pure -- no index to build, persist or invalidate.

    Blind spot worth seeing in the comparison: this only matches literal words. A query
    saying "customer swapped SIM in another city" shares almost nothing with "the swap
    location is more than 100 km from the customer's registered home city", so
    exact-wording queries flatter this baseline and paraphrases flatter the embeddings.

    Hint: `TfidfVectorizer(stop_words='english')`, `fit_transform()`, `transform()`,
          `cosine_similarity()`, `argsort()`
    """
    raise NotImplementedError
