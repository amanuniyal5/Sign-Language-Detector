#!/bin/bash

# ASL Sign Language Detector - Quick Setup Script
# This script helps you set up the project for local development

echo "🤟 ASL Sign Language Detector - Setup"
echo "======================================"
echo ""

# Check Python version
echo "📌 Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
echo ""
echo "🔑 Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "⚠️  Please edit .env and add your GEMINI_API_KEY"
else
    echo "ℹ️  .env file already exists"
fi

# Check if API key is set
echo ""
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not set in environment"
    echo "   Get your key from: https://aistudio.google.com/app/apikey"
    echo "   Then run: export GEMINI_API_KEY='your-key-here'"
else
    echo "✅ GEMINI_API_KEY is set"
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Set your GEMINI_API_KEY (if not already done)"
echo "   2. Run: streamlit run app_final.py"
echo "   3. Open http://localhost:8501 in your browser"
echo ""
echo "🚀 Happy coding!"
