import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clean import process_pdf, clean_text, tokenize_text
import unittest

class TestResumeProcessing(unittest.TestCase):
    
    def test_clean_text_removes_urls(self):
        """Test that URLs are removed"""
        text = "Check out https://github.com and www.google.com"
        cleaned = clean_text(text)
        self.assertNotIn("https://github.com", cleaned)
        self.assertNotIn("www.google.com", cleaned)
    
    def test_clean_text_removes_numbers(self):
        """Test that numbers are removed"""
        text = "Python3 experience since 2023 with 500+ projects"
        cleaned = clean_text(text)
        self.assertNotIn("3", cleaned)
        self.assertNotIn("2023", cleaned)
        self.assertNotIn("500", cleaned)
    
    def test_clean_text_keeps_hyphens(self):
        """Test that hyphens in words are preserved"""
        text = "full-stack developer with co-op experience"
        cleaned = clean_text(text)
        self.assertIn("full-stack", cleaned)
        self.assertIn("co-op", cleaned)
    
    def test_clean_text_keeps_apostrophes(self):
        """Test that apostrophes are preserved"""
        text = "don't can't won't it's"
        cleaned = clean_text(text)
        self.assertIn("don't", cleaned)
        self.assertIn("can't", cleaned)
    
    def test_clean_text_lowercase(self):
        """Test that text is converted to lowercase"""
        text = "PYTHON JavaScript SQL"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "python javascript sql")
    
    def test_tokenize_text(self):
        """Test tokenization"""
        text = "python developer with machine learning"
        tokens = tokenize_text(text)
        expected = ['python', 'developer', 'with', 'machine', 'learning']
        self.assertEqual(tokens, expected)

if __name__ == '__main__':
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Process the actual resume
    print("\n" + "="*50)
    print("PROCESSING YOUR RESUME")
    print("="*50 + "\n")
    
    # Get absolute path from current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    resume_path = os.path.join(current_dir, "..", "assets", "resumes", "Resume - Allen Ho copy.docx.pdf")
    
    if os.path.exists(resume_path):
        tokens = process_pdf(resume_path)
        
        print(f"Total tokens extracted: {len(tokens)}\n")
        print(f"First 50 tokens:")
        print(tokens[:50])
        
        print(f"\n\nUnique tokens (first 100):")
        unique_tokens = sorted(set(tokens))[:100]
        print(unique_tokens)
        
        # Common tech keywords to look for
        tech_keywords = ['python', 'java', 'javascript', 'react', 'node', 'sql', 
                        'aws', 'docker', 'kubernetes', 'git', 'api', 'machine', 
                        'learning', 'data', 'developer', 'engineer', 'software']
        
        found_keywords = [kw for kw in tech_keywords if kw in tokens]
        print(f"\n\nTech keywords found in resume:")
        print(found_keywords)
        
    else:
        print(f"Resume not found at: {resume_path}")
        print(f"Checking directory: {os.path.join(current_dir, '..', 'assets', 'resumes')}")
        resumes_dir = os.path.join(current_dir, "..", "assets", "resumes")
        if os.path.exists(resumes_dir):
            print(f"Available files: {os.listdir(resumes_dir)}")