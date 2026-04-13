"""Assembler module for combining processed features into model-ready datasets."""

from src.data.assembler.assemble import assemble_all, process_single_stock

__all__ = ["assemble_all", "process_single_stock"]
