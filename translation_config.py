"""
Translation Configuration
Store your API keys and translation settings here.
"""

import os

# Try to load from Streamlit secrets (for deployment)
try:
    import streamlit as st
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv('GEMINI_API_KEY', ''))
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv('OPENAI_API_KEY', ''))
except:
    # Fall back to environment variables (for local development)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Choose which API to use: 'gemini' or 'openai'
TRANSLATION_API = 'gemini'  # Change to 'openai' if you prefer

# ============================================================
# TRANSLATION SETTINGS
# ============================================================

# Default target language
DEFAULT_TARGET_LANGUAGE = 'Spanish'

# Available languages (you can add more!)
SUPPORTED_LANGUAGES = {
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Hindi': 'hi',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Arabic': 'ar',
    'Russian': 'ru',
    'Portuguese': 'pt',
    'Italian': 'it',
    'Korean': 'ko',
    'Dutch': 'nl',
    'Turkish': 'tr',
    'Polish': 'pl',
    'Swedish': 'sv',
    'Danish': 'da',
    'Finnish': 'fi',
    'Norwegian': 'no',
    'Greek': 'el',
    'Hebrew': 'he',
    'Thai': 'th',
    'Vietnamese': 'vi',
    'Indonesian': 'id',
    'Malay': 'ms',
    'Bengali': 'bn',
    'Urdu': 'ur',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Gujarati': 'gu',
}

# Translation mode
# 'word': Translate each word separately
# 'sentence': Translate complete sentences
# 'context': Use AI to understand context and provide natural translation
TRANSLATION_MODE = 'sentence'

# Enable/disable automatic translation
AUTO_TRANSLATE = True

# Cache translations to save API calls
ENABLE_CACHE = True
CACHE_FILE = 'translation_cache.json'

# Text-to-speech for translated text
TTS_TRANSLATED_TEXT = True

# ============================================================
# API SETTINGS
# ============================================================

# Gemini settings
GEMINI_MODEL = 'gemini-2.5-flash'  # Latest free tier model (works!)

# OpenAI settings
OPENAI_MODEL = 'gpt-3.5-turbo'  # or 'gpt-4' for better quality

# Request timeout (seconds)
API_TIMEOUT = 10

# Max retries on failure
MAX_RETRIES = 3

# ============================================================
# UI SETTINGS
# ============================================================

# Show translation in UI
SHOW_TRANSLATION = True

# Translation text color (BGR format)
TRANSLATION_COLOR = (0, 165, 255)  # Orange

# Translation font scale
TRANSLATION_FONT_SCALE = 0.8

# Translation position offset
TRANSLATION_Y_OFFSET = 40

# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    """Check if configuration is valid"""
    issues = []
    
    if TRANSLATION_API == 'gemini' and not GEMINI_API_KEY:
        issues.append("⚠️  GEMINI_API_KEY not set. Set it with: export GEMINI_API_KEY='your-key'")
    
    if TRANSLATION_API == 'openai' and not OPENAI_API_KEY:
        issues.append("⚠️  OPENAI_API_KEY not set. Set it with: export OPENAI_API_KEY='your-key'")
    
    if DEFAULT_TARGET_LANGUAGE not in SUPPORTED_LANGUAGES:
        issues.append(f"⚠️  Invalid target language: {DEFAULT_TARGET_LANGUAGE}")
    
    return issues

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_language_code(language_name):
    """Get language code from language name"""
    return SUPPORTED_LANGUAGES.get(language_name, 'es')

def get_language_name(language_code):
    """Get language name from language code"""
    for name, code in SUPPORTED_LANGUAGES.items():
        if code == language_code:
            return name
    return 'Spanish'

def list_available_languages():
    """Return list of available languages"""
    return sorted(SUPPORTED_LANGUAGES.keys())

if __name__ == "__main__":
    print("=" * 60)
    print("Translation Configuration Status")
    print("=" * 60)
    print(f"API Provider: {TRANSLATION_API.upper()}")
    print(f"Target Language: {DEFAULT_TARGET_LANGUAGE}")
    print(f"Translation Mode: {TRANSLATION_MODE}")
    print(f"Auto-Translate: {AUTO_TRANSLATE}")
    print(f"Cache Enabled: {ENABLE_CACHE}")
    print()
    
    issues = validate_config()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ Configuration is valid!")
    
    print()
    print(f"📋 Available Languages ({len(SUPPORTED_LANGUAGES)}):")
    for i, lang in enumerate(list_available_languages(), 1):
        print(f"   {i}. {lang}")
