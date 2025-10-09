#!/usr/bin/env python3
"""
Example/Demo script showing how the release notes generator works with co-authors.

This creates a demo output showing how the script handles different contributor scenarios.
"""

def demo_release_notes():
    """Generate demo release notes to show the format and co-author handling."""
    
    # Sample PRs with different contributor scenarios
    sample_prs = [
        {
            'number': 1812,
            'title': '[MISC] Import math module instead of constants to avoid violating gstaichi pure checker',
            'contributors': ['@hughperkins'],  # Single author
            'category': 'Miscellaneous'
        },
        {
            'number': 1810, 
            'title': '[BUG FIX] Fix shadow map not properly rendered for objects far away from floor plane',
            'contributors': ['@duburcqa'],  # Single author
            'category': 'Bug Fixes'
        },
        {
            'number': 1808,
            'title': '[FEAT] Add support for new rendering backend with GPU acceleration',
            'contributors': ['@author1', '@coauthor1', '@coauthor2'],  # Multiple contributors
            'category': 'New Features'
        },
        {
            'number': 1806,
            'title': 'Improve runtime and compile time performance',
            'contributors': ['@YilingQiao', '@duburcqa'],  # Co-authors from real example
            'category': 'Miscellaneous'
        },
        {
            'number': 1804,
            'title': '[BREAKING] Change default behavior for collision detection',
            'contributors': ['@maintainer', 'Jane Doe', '@external-contributor'],  # Mixed format
            'category': 'Behavior Changing'
        }
    ]
    
    # Group by category
    categories = {
        'Behavior Changing': [],
        'New Features': [],
        'Bug Fixes': [], 
        'Miscellaneous': []
    }
    
    for pr in sample_prs:
        categories[pr['category']].append(pr)
    
    # Generate formatted output
    output = []
    output.append("## 0.4.0")
    output.append("")
    output.append("*This is a demo showing how the release notes generator handles co-authors.*")
    output.append("")
    
    for category_name in ['Behavior Changing', 'New Features', 'Bug Fixes', 'Miscellaneous']:
        if categories[category_name]:
            output.append(f"### {category_name}")
            output.append("")
            
            for pr in sorted(categories[category_name], key=lambda x: x['number']):
                contributors_str = ', '.join(pr['contributors'])
                output.append(f"* {pr['title']}. ({contributors_str}) (#{pr['number']})")
            
            output.append("")
    
    output.append("### Co-Author Benefits")
    output.append("")
    output.append("Notice how the script:")
    output.append("* Lists all contributors, not just the PR author")
    output.append("* Handles both GitHub usernames (@username) and full names")
    output.append("* Preserves the existing release note format")
    output.append("* Removes duplicate contributors within the same PR")
    output.append("* Automatically extracts co-authors from Git commit trailers")
    output.append("")
    
    return '\n'.join(output)


if __name__ == '__main__':
    print(demo_release_notes())