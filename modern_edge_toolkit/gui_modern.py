"""Streamlit user interface for the Modern Edge Detection Toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import streamlit as st

from detectors_modern import (
    SUPPORTED_METHODS,
    list_input_images,
    load_config,
    process_images,
)


def main() -> None:
    st.title("Modern Edge Detection Toolkit")

    cfg = load_config()

    input_dir = st.text_input("Bildordner", cfg.get("input_dir", "images"))
    selected: Sequence[str] = st.multiselect("Methoden", list(SUPPORTED_METHODS))

    start = st.button("🚀 Verarbeitung starten")

    if start:
        images = list_input_images(input_dir)
        if not images:
            st.error("Keine Bilder gefunden")
            return
        progress = st.progress(0.0)

        def update(val: float) -> None:
            progress.progress(val)

        results = process_images(images, selected, cfg, progress_callback=update)
        st.success(f"Fertig! {len(results)} Ergebnisse gespeichert.")
        for res in results:
            st.write(str(res))


if __name__ == "__main__":
    main()
