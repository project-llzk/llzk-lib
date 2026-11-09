#!/usr/bin/env python3
"""Append an operation fragment to a single explicit module in a lit fixture.

The generic module may be original input or printed pass output. Keep trailing
CHECK comments outside the closing brace, and reject an unexpected wrapper.
"""
from pathlib import Path
import sys

module = Path(sys.argv[1]).read_text().splitlines()
body = Path(sys.argv[2]).read_text().splitlines()
content = [i for i, line in enumerate(module) if line.strip() and not line.lstrip().startswith('//')]
if not content or not module[content[0]].lstrip().startswith('module ') or module[content[-1]].strip() != '}':
    raise ValueError('expected one explicit module with a closing brace')
closing = content[-1]
print('\n'.join(module[:closing] + body + module[closing:]))
