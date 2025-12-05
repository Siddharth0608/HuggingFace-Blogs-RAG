"""
Simple text cleaning for HuggingFace blog articles.
Cleans article text and extracts publication date only.
"""

import re
import json
import unicodedata
from typing import Dict, List

# ------------------ TEXT CLEANING ------------------

class ArticleCleaner:
    """Clean article text for storage"""
    
    @staticmethod
    def extract_date(text: str) -> str:
        """Extract publication date from article header"""
        pattern = re.compile(
            r'Published\s+(.*?)(?:\s+Update on GitHub)?\n',
            re.MULTILINE
        )
        match = pattern.search(text)
        return match.group(1).strip() if match else ""
    
    @staticmethod
    def remove_header(text: str) -> str:
        """Remove title and published date from article"""
        pattern = re.compile(
            r'^(.*?)\n+Published\s+(.*?)(?:\s+Update on GitHub)?\n',
            re.DOTALL | re.MULTILINE
        )
        match = pattern.search(text)
        if match:
            return text[match.end():].strip()
        return text
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters (fancy quotes, dashes, etc.)"""
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize excessive whitespace and newlines"""
        # Replace multiple newlines with double newline (paragraph separator)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        return text.strip()
    
    @staticmethod
    def remove_control_chars(text: str) -> str:
        """Remove control characters and invisible unicode"""
        # Remove control characters but keep newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text
    
    @staticmethod
    def clean_markdown_artifacts(text: str) -> str:
        """Clean up common markdown artifacts while preserving structure"""
        # Remove excessive markdown header markers (### becomes regular text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Clean up bold/italic markers but preserve the text
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
        text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
        
        return text
    
    @staticmethod
    def clean_article(text: str, extract_date: bool = True) -> tuple:
        """
        Complete cleaning pipeline for article text.
        
        Args:
            text: Raw article text
            extract_date: Whether to extract and return the date
            
        Returns:
            (cleaned_text, date) if extract_date=True, else (cleaned_text, None)
        """
        date = ""
        if extract_date:
            date = ArticleCleaner.extract_date(text)
        
        # Remove header (title + date line)
        cleaned = ArticleCleaner.remove_header(text)
        
        # Normalize unicode
        cleaned = ArticleCleaner.normalize_unicode(cleaned)
        
        # Remove control characters
        cleaned = ArticleCleaner.remove_control_chars(cleaned)
        
        # Clean markdown artifacts
        cleaned = ArticleCleaner.clean_markdown_artifacts(cleaned)
        
        # Normalize whitespace
        cleaned = ArticleCleaner.normalize_whitespace(cleaned)
        
        return cleaned, date


# ------------------ DATASET PROCESSOR ------------------

def clean_dataset(input_file: str, output_file: str, backup: bool = True):
    """
    Load dataset, clean all articles, and save back to JSON.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (can be same as input)
        backup: Whether to create a backup of the original file
    """
    # Load data
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles")
    
    # Create backup if requested
    if backup and input_file == output_file:
        backup_file = input_file.replace('.json', '_backup.json')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)
        print(f"Created backup at {backup_file}")
    
    # Process each article
    stats = {
        "total": len(articles),
        "cleaned": 0,
        "date_extracted": 0,
        "skipped_errors": 0,
        "skipped_no_text": 0
    }
    
    for article in articles:
        # Skip if not a valid dict or has error
        if not isinstance(article, dict):
            stats["skipped_errors"] += 1
            continue
        
        # Skip if has error flag
        if article.get("_error"):
            stats["skipped_errors"] += 1
            continue
        
        # Skip if no text
        text = article.get("Text", "")
        if not text:
            stats["skipped_no_text"] += 1
            continue
        
        # Clean the article text
        cleaned_text, date = ArticleCleaner.clean_article(text, extract_date=True)
        
        # Update article
        article["Text"] = cleaned_text
        
        if date:
            article["Publish Date"] = date
            stats["date_extracted"] += 1
        
        stats["cleaned"] += 1
    
    # Save cleaned dataset
    print(f"\nSaving cleaned dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Total articles: {stats['total']}")
    print(f"Successfully cleaned: {stats['cleaned']}")
    print(f"Dates extracted: {stats['date_extracted']}")
    print(f"Skipped (errors): {stats['skipped_errors']}")
    print(f"Skipped (no text): {stats['skipped_no_text']}")
    print(f"\nSaved to: {output_file}")
    print("=" * 60)
    
    return stats


# ------------------ TESTING & USAGE ------------------

if __name__ == "__main__":
    clean_dataset(input_file= "Dataset/hf_blogs_data.json", output_file="Dataset/hf_blogs_data.json", backup=True)