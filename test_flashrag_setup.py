#!/usr/bin/env python3
"""
Quick test script to verify FlashRAG setup before using the UI.
This checks if all required components are available.
"""

import sys
import os

def check_imports():
    """Check if required packages are installed."""
    print("Checking Python packages...")
    required_packages = [
        ('gradio', 'Gradio'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name} installed")
        except ImportError:
            print(f"  ❌ {name} NOT installed")
            missing.append(package)
    
    return len(missing) == 0

def check_flashrag():
    """Check if FlashRAG is installed."""
    print("\nChecking FlashRAG installation...")
    try:
        from flashrag.config import Config
        print("  ✅ FlashRAG installed correctly")
        return True
    except ImportError as e:
        print(f"  ❌ FlashRAG not installed: {e}")
        print("  💡 Run: pip install -e src/rag/FlashRAG")
        return False

def check_faiss():
    """Check if FAISS is available."""
    print("\nChecking FAISS...")
    try:
        import faiss
        print(f"  ✅ FAISS installed (version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'})")
        return True
    except ImportError:
        print("  ❌ FAISS not installed")
        print("  💡 Install with: conda install -c pytorch faiss-cpu=1.8.0")
        return False

def check_example_files():
    """Check if example files exist."""
    print("\nChecking example files...")
    flashrag_root = os.path.join(os.path.dirname(__file__), 'src', 'rag', 'FlashRAG')
    
    corpus_path = os.path.join(flashrag_root, 'examples', 'quick_start', 'indexes', 'general_knowledge.jsonl')
    index_path = os.path.join(flashrag_root, 'examples', 'quick_start', 'indexes', 'e5_Flat.index')
    
    files_ok = True
    if os.path.exists(corpus_path):
        size = os.path.getsize(corpus_path) / (1024 * 1024)  # MB
        print(f"  ✅ Corpus file exists: {corpus_path} ({size:.1f} MB)")
    else:
        print(f"  ❌ Corpus file missing: {corpus_path}")
        files_ok = False
    
    if os.path.exists(index_path):
        size = os.path.getsize(index_path) / (1024 * 1024)  # MB
        print(f"  ✅ Index file exists: {index_path} ({size:.1f} MB)")
    else:
        print(f"  ❌ Index file missing: {index_path}")
        files_ok = False
    
    return files_ok

def check_ui_files():
    """Check if UI files are accessible."""
    print("\nChecking UI files...")
    flashrag_root = os.path.join(os.path.dirname(__file__), 'src', 'rag', 'FlashRAG')
    ui_path = os.path.join(flashrag_root, 'webui', 'interface.py')
    
    if os.path.exists(ui_path):
        print(f"  ✅ UI interface file exists: {ui_path}")
        return True
    else:
        print(f"  ❌ UI interface file missing: {ui_path}")
        return False

def main():
    print("=" * 60)
    print("FlashRAG Setup Verification")
    print("=" * 60)
    
    results = {
        'imports': check_imports(),
        'flashrag': check_flashrag(),
        'faiss': check_faiss(),
        'example_files': check_example_files(),
        'ui_files': check_ui_files(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_ok = all(results.values())
    
    for name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name.replace('_', ' ').title()}")
    
    if all_ok:
        print("\n🎉 All checks passed! You're ready to use FlashRAG UI.")
        print("\nTo start the UI, run:")
        print("  cd src/rag/FlashRAG/webui")
        print("  python interface.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nSee INSTALLATION.md for detailed installation instructions.")
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())

