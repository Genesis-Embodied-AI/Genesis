#!/usr/bin/env python3
"""
Release Note Generator with Co-Authors Support

This script generates release notes that include all contributors of merged PRs,
including co-authors specified in commit messages.

Usage:
    python scripts/generate_release_notes.py --since v0.3.3 --until HEAD
    python scripts/generate_release_notes.py --since-date "2024-01-01" --until-date "2024-12-31"
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple


def run_git_command(cmd: List[str]) -> str:
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd="."
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def extract_pr_number(commit_message: str) -> int:
    """Extract PR number from commit message."""
    # Look for (#1234) pattern at the end of the subject line
    match = re.search(r'\(#(\d+)\)$', commit_message.split('\n')[0])
    if match:
        return int(match.group(1))
    return None


def extract_co_authors(commit_message: str) -> List[str]:
    """Extract co-authors from commit message trailers."""
    co_authors = []
    lines = commit_message.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for "Co-authored-by: Name <email>" pattern
        match = re.match(r'^Co-authored-by:\s*([^<]+)<([^>]+)>', line, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            email = match.group(2).strip()
            co_authors.append(f"{name} <{email}>")
    
    return co_authors


def get_github_username_from_email(email: str) -> str:
    """
    Extract GitHub username from email or return the email if not a GitHub email.
    GitHub uses patterns like: username@users.noreply.github.com
    """
    if '@users.noreply.github.com' in email:
        # Extract username from GitHub noreply email
        username = email.split('@')[0]
        # Handle numeric usernames like "12345+username@users.noreply.github.com"
        if '+' in username:
            username = username.split('+')[1]
        return f"@{username}"
    elif email == "noreply@github.com":
        return None  # GitHub web interface, can't determine username
    else:
        # For other emails, we could try to map them, but for now just use the name
        return None


def get_merged_prs_since(since_ref: str, until_ref: str = "HEAD") -> List[Dict]:
    """Get all merged PRs since a given reference."""
    # Get all merge commits since the reference
    cmd = ["git", "log", "--merges", "--oneline", "--format=%H", f"{since_ref}..{until_ref}"]
    merge_commits = run_git_command(cmd).split('\n')
    
    if not merge_commits or merge_commits == ['']:
        return []
    
    prs = []
    
    for commit_hash in merge_commits:
        if not commit_hash:
            continue
            
        # Get commit details
        cmd = ["git", "show", "--format=fuller", "--no-patch", commit_hash]
        commit_info = run_git_command(cmd)
        
        # Extract commit message
        lines = commit_info.split('\n')
        message_start = None
        for i, line in enumerate(lines):
            if line.strip() == '' and i > 0:
                message_start = i + 1
                break
        
        if message_start is None:
            continue
            
        commit_message = '\n'.join(lines[message_start:])
        subject_line = commit_message.split('\n')[0]
        
        # Extract PR number
        pr_number = extract_pr_number(subject_line)
        if pr_number is None:
            continue
        
        # Extract author info
        author_line = next((line for line in lines if line.startswith('Author:')), '')
        author_match = re.match(r'Author:\s+([^<]+)<([^>]+)>', author_line)
        
        if not author_match:
            continue
            
        author_name = author_match.group(1).strip()
        author_email = author_match.group(2).strip()
        
        # Extract co-authors
        co_authors = extract_co_authors(commit_message)
        
        # Get all contributors (author + co-authors)
        contributors = []
        
        # Add main author
        github_username = get_github_username_from_email(author_email)
        if github_username:
            contributors.append(github_username)
        else:
            contributors.append(f"{author_name}")
        
        # Add co-authors
        for co_author in co_authors:
            # Parse co-author string "Name <email>"
            co_match = re.match(r'^([^<]+)<([^>]+)>$', co_author)
            if co_match:
                co_name = co_match.group(1).strip()
                co_email = co_match.group(2).strip()
                co_github_username = get_github_username_from_email(co_email)
                if co_github_username:
                    contributors.append(co_github_username)
                else:
                    contributors.append(co_name)
        
        prs.append({
            'number': pr_number,
            'title': subject_line.replace(f' (#{pr_number})', ''),
            'contributors': contributors,
            'commit_hash': commit_hash,
            'commit_message': commit_message
        })
    
    return prs


def categorize_pr(title: str) -> str:
    """Categorize PR based on its title."""
    title_lower = title.lower()
    
    if any(keyword in title_lower for keyword in ['[bug fix]', 'fix', 'bugfix', 'bug']):
        return 'Bug Fixes'
    elif any(keyword in title_lower for keyword in ['[feat]', '[feature]', 'add', 'implement', 'support']):
        return 'New Features'
    elif any(keyword in title_lower for keyword in ['breaking', 'behavior']):
        return 'Behavior Changing'
    else:
        return 'Miscellaneous'


def format_contributors(contributors: List[str]) -> str:
    """Format contributors list for release notes."""
    # Remove duplicates while preserving order
    unique_contributors = []
    seen = set()
    for contributor in contributors:
        if contributor not in seen:
            unique_contributors.append(contributor)
            seen.add(contributor)
    
    return ', '.join(unique_contributors)


def generate_release_notes(prs: List[Dict], version: str = None) -> str:
    """Generate formatted release notes."""
    if not prs:
        return "No merged PRs found in the specified range."
    
    # Group PRs by category
    categorized_prs = defaultdict(list)
    for pr in prs:
        category = categorize_pr(pr['title'])
        categorized_prs[category].append(pr)
    
    # Generate release notes
    notes = []
    
    if version:
        notes.append(f"## {version}")
        notes.append("")
    
    # Define category order
    category_order = ['Behavior Changing', 'New Features', 'Bug Fixes', 'Miscellaneous']
    
    for category in category_order:
        if category in categorized_prs:
            notes.append(f"### {category}")
            notes.append("")
            
            for pr in sorted(categorized_prs[category], key=lambda x: x['number']):
                contributors_str = format_contributors(pr['contributors'])
                notes.append(f"* {pr['title']}. ({contributors_str}) (#{pr['number']})")
            
            notes.append("")
    
    return '\n'.join(notes)


def main():
    parser = argparse.ArgumentParser(description='Generate release notes with co-author support')
    parser.add_argument('--since', help='Starting git reference (tag, commit, etc.)')
    parser.add_argument('--until', default='HEAD', help='Ending git reference (default: HEAD)')
    parser.add_argument('--since-date', help='Starting date (YYYY-MM-DD)')
    parser.add_argument('--until-date', help='Ending date (YYYY-MM-DD)')
    parser.add_argument('--version', help='Version number for the release notes')
    parser.add_argument('--output', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    if not args.since and not args.since_date:
        parser.error('Must specify either --since or --since-date')
    
    # Determine the git range
    if args.since_date:
        since_ref = f"--since='{args.since_date}'"
        until_ref = f"--until='{args.until_date}'" if args.until_date else "HEAD"
        
        # Get the first commit in the date range
        cmd = ["git", "log", "--reverse", "--format=%H", since_ref]
        if args.until_date:
            cmd.append(until_ref)
        first_commit = run_git_command(cmd).split('\n')[0]
        
        if not first_commit:
            print("No commits found in the specified date range.")
            sys.exit(1)
        
        since_ref = f"{first_commit}^"
        until_ref = args.until
    else:
        since_ref = args.since
        until_ref = args.until
    
    # Get merged PRs
    print(f"Analyzing merged PRs from {since_ref} to {until_ref}...", file=sys.stderr)
    prs = get_merged_prs_since(since_ref, until_ref)
    
    print(f"Found {len(prs)} merged PRs", file=sys.stderr)
    
    # Generate release notes
    release_notes = generate_release_notes(prs, args.version)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(release_notes)
        print(f"Release notes written to {args.output}", file=sys.stderr)
    else:
        print(release_notes)


if __name__ == '__main__':
    main()