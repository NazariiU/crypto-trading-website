import streamlit as st
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the web application
from web_app import main

# Run the main function
if __name__ == "__main__":
    main()
