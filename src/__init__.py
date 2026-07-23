"""src — Clinic RAG Bot source package."""
import os
import sys

# Add this directory to sys.path to enable local sibling imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

