#!/usr/bin/env python3
# One-liner approach - paste your base64 string here:
base64_string = """b'iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAEAAElEQVR4nNz9a5MkOY4siKoCNPfIqu45+///4X2I7JnuqsoMNxK4H/AgPTK7z4rsXpE5EzOdlelhbkYjAYXiQRAAQIDxHwDM/wL9K+Lrb3h+ob9HgvFv7s9RnzD+SrJ+x/wW61d5gfQD84P3IRHnv3n+/fgTv77k7YKfrtwDOz54e8BxAWviajp4fpXs577PRt4k35ykiJByPqfu3DMSk8S4kNI35i9e4eefPS17yF8n4/29j/fbP6cE/OqW/fo8H/N2z68r8T6QnJiWjfok7vnTyM/vikh9K++1R/OvH/n29Hob1r96dnsZ9ntIC20Obst+vWh8xLy4X2x/dw+Ue9z9sJ4T7PG0XCDf99++T//nX0/bV5X58l+GEG+NPvQ23/6LqpyS+kttJMGeARGKClUgp6TndW8D+rL6fP//L2/9r162vshElS8q+vM8/OIe/PU/vg5hA109jsKU4vrzl6OuK6FD+wV/uvRtFF/0r1/peL/84LxRqXR/3eFbfdz7e56XOAkH4CQ9fx8f0d+/DLg7QObHBAB3j5sQDk9p9rgw/pZCVjdyB3E8x/POzCGB9XTPUeYDvWcyPidJd4PDCTpiHCCQHyBuSfSv84sAHL4n0evCGgGQLx7Dj3vlFc5+l5gu1re9Zql+W7N8LGrNW03m8XkuBuN/cAe8J9rf73QIC2uR/RAgxuNriXLxU1D6Tq3oMVQ/BrHFpGZgf6sn7P3Fen5RyuCEW/MEWiyyO0ARuIOkh9BZL3TcqgWE7i5CM39/2vn0Q4ARokTPNfNffuNcjPyyl4QfLxQL7e41UYwZcjtWDiyZ8DchOu5x/GKLZHzzWFgCDqEyXtZTRreUkKE7oQKlYSGevVz79TzGG+Nni2MvZc1WvUe9jqciJ0J8AcMY1i/ms5cPJW7wnlIBIQBoAM1zMOet/X1+WhNL2XOi95oe2MXUdzhYhMGRcFbTT7SqFiqd0/su48"""

import base64

with open("diff_image.png", "wb") as f:
    f.write(base64.b64decode(base64_string.strip().strip("'").strip('"').lstrip("b'").rstrip("'")))
print("Saved to diff_image.png")
