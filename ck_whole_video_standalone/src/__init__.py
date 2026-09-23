"""Minimal package init for the standalone CK whole-video rating bundle.

The full project's __init__ eagerly imports analysis/audio modules that this
bundle does not ship. The two scripts import submodules explicitly
(``from src.ck_config import CKConfig`` etc.), so this file stays empty on
purpose — importing the package must not pull in anything extra.
"""
