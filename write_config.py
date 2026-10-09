"""
write_config.py - put a Gemini key from .env into config.js for the app.

index.html loads config.js, which sets window.BREVIS_SHARED_KEY. The app uses
that key ONLY for books the catalogue does not have: the "Find books like ..."
fallback in More like this, and looking up a book or author it has never
heard of. Everything else is untouched.

    python write_config.py                 # first of: BREVIS_SHARED_KEY,
                                           # GEMINI_API_KEY, GOOGLE_API_KEY
    python write_config.py --var MY_KEY    # use a specific .env variable

READ THIS BEFORE COMMITTING config.js
    config.js is served to every visitor, so the key in it is public: anyone
    can read it and spend its quota. Use a key from a free-tier Google project
    with no billing set up, so the worst case is a used-up quota rather than a
    bill, and replace the key if that happens. GitHub scans public repos for
    Google keys and may block the push or report the key to Google.

The script never prints the key.
"""

import argparse
import json
import os
import sys

PREFERRED = ["BREVIS_SHARED_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]


def read_env(path=".env"):
    values = {}
    if not os.path.exists(path):
        sys.exit("No .env file here. Run this from the Brevis folder.")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def main():
    ap = argparse.ArgumentParser(description="Write config.js from a key in .env.")
    ap.add_argument("--var", help="which .env variable holds the key")
    args = ap.parse_args()

    env = read_env()
    names = [args.var] if args.var else PREFERRED
    name = next((n for n in names if env.get(n)), None)
    if not name:
        sys.exit("None of %s is set in .env. Variables found: %s"
                 % (", ".join(names), ", ".join(sorted(env)) or "none"))

    with open("config.js", "w", encoding="utf-8") as fh:
        fh.write("/* Written by write_config.py from .env. This key is public - see the\n"
                 "   note in write_config.py. Used only for books the catalogue lacks. */\n")
        fh.write("window.BREVIS_SHARED_KEY = %s;\n" % json.dumps(env[name]))

    print("Wrote config.js using %s from .env (the key itself is not shown)." % name)
    print("Next: git add config.js && git commit -m \"Add shared Gemini key\" && git push")


if __name__ == "__main__":
    main()
