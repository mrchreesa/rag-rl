import os
import glob
import json
import random
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from tqdm import tqdm

# 1. Configuration
SOURCE_DIR = "./docs/test"  # Path to test PDFs
OUTPUT_FILE = "./datasets/test_dataset.jsonl"
QA_PER_CHUNK = 2  # QA pairs to generate per chunk
TARGET_QA_PAIRS = 200  # Target number of QA pairs to generate
RANDOM_SEED = 42  # For reproducibility

# 2. Setup Local Model
print("Initializing Local Llama 3.1...")
llm = ChatOllama(model="llama3.1:8b", temperature=0.7)

# 3. Load and Chunk Documents from ALL PDFs
print(f"Loading PDFs from {SOURCE_DIR}...")
pdf_files = glob.glob(os.path.join(SOURCE_DIR, "*.pdf"))
print(f"Found {len(pdf_files)} PDFs - processing ALL for full dataset representation...")
documents = []

for i, file_path in enumerate(pdf_files):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_file'] = os.path.basename(file_path)
        documents.extend(docs)
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(pdf_files)} PDFs loaded ({len(documents)} pages so far)...")
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")

print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDFs")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks from all documents")

# Randomly sample chunks from across the entire dataset for fair representation
chunks_needed = TARGET_QA_PAIRS // QA_PER_CHUNK
random.seed(RANDOM_SEED)
if len(chunks) > chunks_needed:
    chunks_to_process = random.sample(chunks, chunks_needed)
    print(f"Randomly sampled {len(chunks_to_process)} chunks from {len(chunks)} total chunks")
else:
    chunks_to_process = chunks
    print(f"Using all {len(chunks)} chunks (less than target)")

# 4. Generate QA Pairs
def generate_qa_from_chunk(chunk_text: str, source: str) -> list:
    """Generate QA pairs from a text chunk using Ollama."""
    
    prompt = f"""You are an expert at creating educational question-answer pairs.
Based on the following academic text, generate {QA_PER_CHUNK} high-quality question-answer pairs.

Rules:
1. Questions should test understanding of key concepts
2. Answers should be complete and directly based on the text
3. Include a mix of factual and reasoning questions
4. Output ONLY valid JSON array format

Text:
{chunk_text[:2500]}

Output format (JSON array only, no other text):
[
  {{"question": "...", "answer": "...", "type": "factual"}},
  {{"question": "...", "answer": "...", "type": "reasoning"}}
]"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Try to extract JSON from response
        if '[' in content and ']' in content:
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            json_str = content[json_start:json_end]
            qa_pairs = json.loads(json_str)
            
            # Add source metadata
            for qa in qa_pairs:
                qa['source'] = source
                qa['context'] = chunk_text[:500]  # Store partial context
            
            return qa_pairs
        else:
            print(f"    [Warning] No JSON found in response")
    except json.JSONDecodeError as e:
        print(f"    [Warning] JSON parse error: {e}")
    except Exception as e:
        print(f"    [Error] LLM call failed: {e}")
    
    return []

# 5. Process chunks and generate QA pairs
print(f"\nGenerating QA pairs from {len(chunks_to_process)} randomly sampled chunks...")
all_qa_pairs = []

for i, chunk in enumerate(tqdm(chunks_to_process, desc="Processing chunks")):
    chunk_text = chunk.page_content
    source = chunk.metadata.get('source_file', 'unknown')
    
    # Skip very short chunks
    if len(chunk_text) < 200:
        continue
    
    qa_pairs = generate_qa_from_chunk(chunk_text, source)
    all_qa_pairs.extend(qa_pairs)
    
    # Progress update every 10 chunks
    if (i + 1) % 10 == 0:
        print(f"  Generated {len(all_qa_pairs)} QA pairs so far...")

# 6. Save to JSONL (standard format for ML datasets)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

if all_qa_pairs:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for qa in all_qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Saved {len(all_qa_pairs)} QA pairs to {OUTPUT_FILE}")
    print(f"   Format: JSONL (JSON Lines)")
    print(f"\nSample QA pair:")
    print(f"   Q: {all_qa_pairs[0]['question'][:100]}...")
    print(f"   A: {all_qa_pairs[0]['answer'][:100]}...")
else:
    print("\n❌ No QA pairs were generated. Check if Ollama is running.")
