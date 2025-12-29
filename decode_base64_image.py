#!/usr/bin/env python3
"""
Script to decode base64-encoded PNG image and save it to a file.

Usage:
    python decode_base64_image.py <base64_string> [output_file.png]

Example:
    python decode_base64_image.py "iVBORw0KGgoAAAANSUhEUgAA..." diff_image.png
"""

import base64
import sys


def find_non_ascii(text, max_reports=20):
    """Find positions of non-ASCII characters in text."""
    non_ascii_positions = []
    for i, char in enumerate(text):
        # Check if character is ASCII (ord < 128)
        if ord(char) >= 128:
            non_ascii_positions.append((i, char, ord(char), repr(char)))
    return non_ascii_positions


def decode_and_save(base64_file, output_file="decoded_image.png", max_reports=20):
    """Decode base64 string and save as PNG file."""
    try:
        # Try to open with utf-8-sig first (automatically strips BOM)
        # If that fails, fall back to utf-8
        try:
            with open(base64_file, "r", encoding="utf-8-sig") as f:
                base64_string = f.read()
            # Check if BOM was present (utf-8-sig strips it automatically)
            with open(base64_file, "rb") as f:
                first_bytes = f.read(3)
                if first_bytes == b"\xef\xbb\xbf":
                    print("ℹ️  Detected UTF-8 BOM at start of file (automatically removed)")
        except Exception:
            with open(base64_file, "r", encoding="utf-8") as f:
                base64_string = f.read()

        # Check for non-ASCII characters
        non_ascii = find_non_ascii(base64_string)
        if non_ascii:
            print(f"\n⚠️  Found {len(non_ascii)} non-ASCII character(s) in the file:")
            print("=" * 70)
            for pos, char, code, repr_char in non_ascii[:max_reports]:  # Show first max_reports
                # Special handling for BOM
                char_name = "BOM (Byte Order Mark)" if code == 0xFEFF else "non-ASCII"
                context_start = max(0, pos - 30)
                context_end = min(len(base64_string), pos + 30)
                context = base64_string[context_start:context_end]
                # Highlight the problematic character
                highlight_pos = pos - context_start
                # Show context with the problematic character highlighted
                before = context[:highlight_pos]
                after = context[highlight_pos + 1 :]
                highlighted = f"{before}>>>{repr_char}<<<{after}"
                # Calculate line and column
                line_num = base64_string[:pos].count("\n") + 1
                last_newline = base64_string.rfind("\n", 0, pos)
                col_num = pos - last_newline - 1 if last_newline >= 0 else pos + 1
                print(f"\n  Position {pos} (line {line_num}, column {col_num}):")
                print(f"    Character: {repr_char} ({char_name})")
                print(f"    Unicode: U+{code:04X} (decimal {code})")
                print(f"    Context:   ...{highlighted}...")
            if len(non_ascii) > max_reports:
                print(f"\n  ... and {len(non_ascii) - max_reports} more non-ASCII character(s)")
            print("=" * 70)
            print("\nAttempting to clean non-ASCII characters (removing them)...")

            # Remove non-ASCII characters
            cleaned = "".join(c for c in base64_string if ord(c) < 128)
            removed_count = len(base64_string) - len(cleaned)
            print(f"✓ Removed {removed_count} non-ASCII character(s)")
            base64_string = cleaned

        # Remove any whitespace and quotes if present
        base64_string = base64_string.strip().strip("'").strip('"')

        # Remove 'b' prefix if present (from Python bytes representation)
        if base64_string.startswith("b'"):
            base64_string = base64_string[2:-1]
        elif base64_string.startswith('b"'):
            base64_string = base64_string[2:-1]

        # Decode base64 to bytes
        png_bytes = base64.b64decode(base64_string)

        # Save to file
        with open(output_file, "wb") as f:
            f.write(png_bytes)

        print(f"Successfully decoded and saved to: {output_file}")
        print(f"Image size: {len(png_bytes)} bytes")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    base64_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "decoded_image.png"

    decode_and_save(base64_file, output_file)
