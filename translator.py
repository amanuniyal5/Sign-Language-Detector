"""
AI-Powered Translator Module
Supports Google Gemini and OpenAI APIs for translation
"""

import json
import os
import time
from typing import Optional, Dict
import translation_config as config

# Try to import required libraries
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-genai not installed. Install with: pip install google-genai")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai not installed. Install with: pip install openai")


class Translator:
    """
    AI-powered translator using Gemini or OpenAI APIs
    """
    
    def __init__(self, api_provider=None, target_language=None):
        """
        Initialize translator
        
        Args:
            api_provider: 'gemini' or 'openai' (defaults to config)
            target_language: Target language name (defaults to config)
        """
        self.api_provider = api_provider or config.TRANSLATION_API
        self.target_language = target_language or config.DEFAULT_TARGET_LANGUAGE
        self.translation_cache = {}
        self.last_request_time = 0
        self.request_count = 0
        
        # Load cache
        if config.ENABLE_CACHE:
            self._load_cache()
        
        # Initialize API
        self._initialize_api()
    
    def _initialize_api(self):
        """Initialize the selected API"""
        if self.api_provider == 'gemini':
            if not GEMINI_AVAILABLE:
                print("❌ Gemini API not available. Install: pip install google-genai")
                self.api_initialized = False
                return
            
            if not config.GEMINI_API_KEY:
                print("❌ GEMINI_API_KEY not set!")
                print("   Get your key from: https://makersuite.google.com/app/apikey")
                print("   Set it with: export GEMINI_API_KEY='your-key'")
                self.api_initialized = False
                return
            
            try:
                self.client = genai.Client(api_key=config.GEMINI_API_KEY)
                self.api_initialized = True
                print(f"✅ Gemini API initialized (gemini-2.5-flash)")
            except Exception as e:
                print(f"❌ Failed to initialize Gemini: {e}")
                self.api_initialized = False
        
        elif self.api_provider == 'openai':
            if not OPENAI_AVAILABLE:
                print("❌ OpenAI API not available. Install: pip install openai")
                self.api_initialized = False
                return
            
            if not config.OPENAI_API_KEY:
                print("❌ OPENAI_API_KEY not set!")
                print("   Get your key from: https://platform.openai.com/api-keys")
                print("   Set it with: export OPENAI_API_KEY='your-key'")
                self.api_initialized = False
                return
            
            try:
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                self.api_initialized = True
                print(f"✅ OpenAI API initialized ({config.OPENAI_MODEL})")
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI: {e}")
                self.api_initialized = False
        
        else:
            print(f"❌ Unknown API provider: {self.api_provider}")
            self.api_initialized = False
    
    def _load_cache(self):
        """Load translation cache from file"""
        if os.path.exists(config.CACHE_FILE):
            try:
                with open(config.CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.translation_cache = json.load(f)
                print(f"📦 Loaded {len(self.translation_cache)} cached translations")
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
                self.translation_cache = {}
    
    def _save_cache(self):
        """Save translation cache to file"""
        if config.ENABLE_CACHE:
            try:
                with open(config.CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️  Failed to save cache: {e}")
    
    def _get_cache_key(self, text, target_lang):
        """Generate cache key"""
        return f"{text.lower()}:{target_lang}"
    
    def translate(self, text: str, target_language: Optional[str] = None) -> Optional[str]:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language (defaults to configured language)
        
        Returns:
            Translated text or None if translation failed
        """
        if not text or not text.strip():
            return None
        
        if not self.api_initialized:
            return None
        
        target_lang = target_language or self.target_language
        
        # Check cache first
        cache_key = self._get_cache_key(text, target_lang)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < 1:  # 1 second between requests
            time.sleep(1 - (current_time - self.last_request_time))
        
        # Translate using selected API
        try:
            if self.api_provider == 'gemini':
                translated = self._translate_gemini(text, target_lang)
            elif self.api_provider == 'openai':
                translated = self._translate_openai(text, target_lang)
            else:
                translated = None
            
            # Cache the result
            if translated and config.ENABLE_CACHE:
                self.translation_cache[cache_key] = translated
                self._save_cache()
            
            self.last_request_time = time.time()
            self.request_count += 1
            
            return translated
        
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            return None
    
    def _translate_gemini(self, text: str, target_lang: str) -> Optional[str]:
        """Translate using Google Gemini"""
        prompt = self._create_translation_prompt(text, target_lang)
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            translated = response.text.strip()
            return translated
        except Exception as e:
            print(f"Gemini translation error: {e}")
            return None
    
    def _translate_openai(self, text: str, target_lang: str) -> Optional[str]:
        """Translate using OpenAI"""
        prompt = self._create_translation_prompt(text, target_lang)
        
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Provide only the translation, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            translated = response.choices[0].message.content.strip()
            return translated
        except Exception as e:
            print(f"OpenAI translation error: {e}")
            return None
    
    def _create_translation_prompt(self, text: str, target_lang: str) -> str:
        """Create translation prompt based on mode"""
        if config.TRANSLATION_MODE == 'word':
            prompt = f"Translate the word '{text}' to {target_lang}. Provide only the translation, nothing else."
        
        elif config.TRANSLATION_MODE == 'sentence':
            prompt = f"Translate this text to {target_lang}: '{text}'\n\nProvide only the translation, no explanations."
        
        elif config.TRANSLATION_MODE == 'context':
            prompt = f"""Translate this sign language text to natural {target_lang}: '{text}'

This text was detected from sign language gestures. Provide a natural, contextually appropriate translation.
Respond with ONLY the translation, no explanations."""
        
        else:
            prompt = f"Translate to {target_lang}: {text}"
        
        return prompt
    
    def set_target_language(self, language: str):
        """Change target language"""
        if language in config.SUPPORTED_LANGUAGES:
            self.target_language = language
            print(f"🌍 Target language changed to: {language}")
        else:
            print(f"⚠️  Language not supported: {language}")
    
    def get_stats(self) -> Dict:
        """Get translation statistics"""
        return {
            'api_provider': self.api_provider,
            'target_language': self.target_language,
            'request_count': self.request_count,
            'cache_size': len(self.translation_cache),
            'api_initialized': self.api_initialized
        }
    
    def clear_cache(self):
        """Clear translation cache"""
        self.translation_cache = {}
        if os.path.exists(config.CACHE_FILE):
            os.remove(config.CACHE_FILE)
        print("🗑️  Translation cache cleared")


# Quick test function
if __name__ == "__main__":
    print("=" * 60)
    print("Translator Test")
    print("=" * 60)
    
    # Create translator
    translator = Translator()
    
    if not translator.api_initialized:
        print("\n❌ Translator not initialized. Please configure API keys.")
        print("\nSetup Instructions:")
        print("1. Get Gemini API key: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable: export GEMINI_API_KEY='your-key'")
        print("3. Run this script again")
    else:
        # Test translations
        test_phrases = [
            "HELLO",
            "GOOD MORNING",
            "THANK YOU",
            "HOW ARE YOU",
        ]
        
        print(f"\nTranslating to {translator.target_language}...\n")
        
        for phrase in test_phrases:
            translated = translator.translate(phrase)
            if translated:
                print(f"✅ {phrase:20s} → {translated}")
            else:
                print(f"❌ {phrase:20s} → Translation failed")
        
        # Show stats
        print("\n" + "=" * 60)
        stats = translator.get_stats()
        print("Translation Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
