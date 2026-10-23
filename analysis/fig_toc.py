"""Table-of-contents graphic (3.25 in x 1.75 in): the graphical-abstract cards with shorter wording."""
import importlib

import figstyle as fs

ga = importlib.import_module("fig0_graphical_abstract")
S = fs.Sheet(1.75 * 25.4, width_mm=3.25 * 25.4)
cards = ga.ga_content(S, 1.0, 1.0, 3.25 * 25.4 - 2.0, 1.75 * 25.4 - 2.0, compact=True)
fs.export(S.fig, "TOC_graphic", axes=cards, ids=["a", "b", "c"], row_groups=[["a", "b", "c"]])
