#!/usr/bin/env python3
"""
Test script for the release notes generator.
Creates a sample output based on the actual Genesis repository structure.
"""

import sys
import os

# Add the scripts directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_github_username_extraction():
    """Test the GitHub username extraction functionality."""
    from generate_release_notes import get_github_username_from_email
    
    test_cases = [
        ("12345+hughperkins@users.noreply.github.com", "@hughperkins"),
        ("jane@users.noreply.github.com", "@jane"),
        ("noreply@github.com", None),
        ("user@example.com", None),
        ("123456789+duburcqa@users.noreply.github.com", "@duburcqa"),
    ]
    
    print("Testing GitHub username extraction:")
    for email, expected in test_cases:
        result = get_github_username_from_email(email)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {email} -> {result} (expected: {expected})")
    print()


def test_co_author_extraction():
    """Test the co-author extraction functionality."""
    from generate_release_notes import extract_co_authors
    
    sample_commit_message = """[MISC] Import math module instead of constants (#1812)

This change helps avoid violating the gstaichi pure checker by importing
the entire math module instead of specific constants.

Co-authored-by: Jane Doe <jane@example.com>
Co-authored-by: John Smith <12345+johnsmith@users.noreply.github.com>
Co-authored-by: Alice Johnson <alice@users.noreply.github.com>"""
    
    co_authors = extract_co_authors(sample_commit_message)
    
    print("Testing co-author extraction:")
    print(f"  Found {len(co_authors)} co-authors:")
    for i, author in enumerate(co_authors, 1):
        print(f"    {i}. {author}")
    print()


def test_pr_categorization():
    """Test the PR categorization functionality."""
    from generate_release_notes import categorize_pr
    
    test_cases = [
        ("[BUG FIX] Fix shadow map rendering", "Bug Fixes"),
        ("[FEAT] Add new rendering backend", "New Features"),
        ("[MISC] Update documentation", "Miscellaneous"),
        ("Improve runtime performance", "Miscellaneous"),
        ("Fix critical bug in collision detection", "Bug Fixes"),
        ("Add support for new file format", "New Features"),
        ("[BREAKING] Change default behavior", "Behavior Changing"),
    ]
    
    print("Testing PR categorization:")
    for title, expected in test_cases:
        result = categorize_pr(title)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{title}' -> {result}")
    print()


def create_sample_release_notes():
    """Create sample release notes showing the expected format."""
    from generate_release_notes import generate_release_notes
    
    # Sample PRs with different contributor patterns
    sample_prs = [
        {
            'number': 1812,
            'title': '[MISC] Import math module instead of constants to avoid violating gstaichi pure checker',
            'contributors': ['@hughperkins'],
            'commit_hash': 'a73bda7',
            'commit_message': 'Sample commit'
        },
        {
            'number': 1664,
            'title': '[BUG FIX] Fix shadow map not properly rendered for objects far away from floor plane',
            'contributors': ['@duburcqa'],
            'commit_hash': 'abc1234',
            'commit_message': 'Sample commit'
        },
        {
            'number': 1512,
            'title': 'Fix point-cloud rendering from Camera depth map',
            'contributors': ['@@ceasor-mao', '@duburcqa'],  # Note: real example from RELEASE.md has @@
            'commit_hash': 'def5678',
            'commit_message': 'Sample commit'
        },
        {
            'number': 1164,
            'title': 'Improve runtime and compile time performance',
            'contributors': ['@YilingQiao', '@duburcqa'],
            'commit_hash': 'ghi9012',
            'commit_message': 'Sample commit'
        }
    ]
    
    print("Sample release notes output:")
    print("=" * 50)
    release_notes = generate_release_notes(sample_prs, "0.4.0")
    print(release_notes)


if __name__ == '__main__':
    print("Release Notes Generator - Test Suite")
    print("=" * 50)
    print()
    
    test_github_username_extraction()
    test_co_author_extraction() 
    test_pr_categorization()
    create_sample_release_notes()