import os
import sys

# Ensure root folder is in Python search path for Vercel Serverless Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Export handler for Vercel
app = app
