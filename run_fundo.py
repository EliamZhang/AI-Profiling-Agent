#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fundo 产品画像入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from generate_report import main

PRODUCT_CONFIG = Path(__file__).resolve().parent / "products" / "fundo.json"


if __name__ == "__main__":
    raise SystemExit(main(["--config", str(PRODUCT_CONFIG), *sys.argv[1:]]))
