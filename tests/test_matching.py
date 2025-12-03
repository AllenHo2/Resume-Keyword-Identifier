"""
Quick test to verify resume-to-job matching works correctly
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import find_resume_keywords_in_job

# Test data
test_resume_keywords = [
    ('python', 0.5),
    ('machine learning', 0.4),
    ('tensorflow', 0.35),
    ('docker', 0.3),
    ('react', 0.25),
    ('mongodb', 0.2)
]

test_job_text = """
We are seeking a Software Engineer with strong Python programming skills.
Experience with machine learning and TensorFlow is required.
Familiarity with Docker and containerization is a plus.
Knowledge of AWS cloud services is preferred.
"""

print("Testing Resume-to-Job Keyword Matching")
print("=" * 60)

found, missing = find_resume_keywords_in_job(test_resume_keywords, test_job_text)

print(f"\nResume Keywords: {len(test_resume_keywords)}")
print(f"Found in Job: {len(found)}")
print(f"Missing from Job: {len(missing)}")

print("\n✅ Found Keywords:")
for keyword, score in found:
    print(f"   - {keyword} (score: {score})")

print("\n❌ Missing Keywords:")
for keyword, score in missing:
    print(f"   - {keyword} (score: {score})")

match_rate = len(found) / len(test_resume_keywords) * 100
print(f"\n📊 Match Rate: {match_rate:.1f}%")

# Expected results
expected_found = ['python', 'machine learning', 'tensorflow', 'docker']
expected_missing = ['react', 'mongodb']

actual_found = [k for k, _ in found]
actual_missing = [k for k, _ in missing]

if set(actual_found) == set(expected_found) and set(actual_missing) == set(expected_missing):
    print("\n✓ TEST PASSED: Matching works correctly!")
else:
    print("\n✗ TEST FAILED: Unexpected results")
    print(f"   Expected found: {expected_found}")
    print(f"   Actual found: {actual_found}")
    print(f"   Expected missing: {expected_missing}")
    print(f"   Actual missing: {actual_missing}")
