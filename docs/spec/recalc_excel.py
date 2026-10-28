"""Recalculate a workbook in Excel and save it, so that its cached values can be read back.

openpyxl writes formulas without cached results. The reference workbooks (D22) are built with
openpyxl from the product specifications, and the engine's reconciliation tests read their values
with ``openpyxl.load_workbook(path, data_only=True)``, which needs the values Excel computed.
This tool opens each workbook in Excel through COM (pywin32, Windows only), forces a full
recalculation, saves, and reports the value of the workbook's ``all_ok`` cell (its defined name,
or a cell labelled ``all_ok`` with the value to its right) when there is one.

    uv run --frozen python docs/spec/recalc_excel.py docs/spec/term-assurance/term-assurance.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path


def recalc(path: Path) -> str | None:
    """Recalculate ``path`` in place and return the ``all_ok`` cell as text when present."""
    import pythoncom  # pyright: ignore[reportMissingModuleSource]
    import win32com.client  # pyright: ignore[reportMissingModuleSource]

    pythoncom.CoInitialize()
    app = win32com.client.DispatchEx("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    try:
        wb = app.Workbooks.Open(str(path.resolve()))
        try:
            app.CalculateFullRebuild()
            all_ok: str | None = None
            for name in wb.Names:  # the workbook-level defined name, wherever the cell sits
                if name.Name.split("!")[-1] == "all_ok":
                    all_ok = str(name.RefersToRange.Value)
            if all_ok is None:
                for sheet in wb.Worksheets:
                    found = sheet.Cells.Find("all_ok")
                    if found is not None:
                        all_ok = str(found.Offset(1, 2).Value)
                        break
            wb.Save()
            return all_ok
        finally:
            wb.Close(SaveChanges=False)
    finally:
        app.Quit()
        pythoncom.CoUninitialize()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        path = Path(arg)
        if not path.exists():
            print(f"{path}: no such file")
            return 1
        all_ok = recalc(path)
        suffix = f", all_ok = {all_ok}" if all_ok is not None else ""
        print(f"{path}: recalculated{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
