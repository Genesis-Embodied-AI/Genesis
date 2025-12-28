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


def decode_and_save(base64_string, output_file="decoded_image.png"):
    """Decode base64 string and save as PNG file."""
    try:
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

    base64_string = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "decoded_image.png"

    decode_and_save(base64_string, output_file)
