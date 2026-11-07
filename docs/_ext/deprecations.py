# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An MkDocs hook that writes the current deprecations into
``reference/deprecations.md`` from Tack's registry (``tack._deprecation``), where
the page says ``<!-- deprecations -->``: what is deprecated, since when, until
when, and what to use, so the page is never behind the code."""

_PAGE = "reference/deprecations.md"
_MARK = "<!-- deprecations -->"


def _table():
    from tack._deprecation import registry

    found = registry()
    if not found:
        return "Nothing is deprecated."
    rows = ["| Deprecated | Package | Since | Removed in | Use instead |",
            "|---|---|---|---|---|"]
    for d in found:
        rows.append(f"| `{d.what}` | {d.package} | {d.since} | {d.removed_in} | "
                    f"{d.use.replace('|', '&#124;')} |")
    return "\n".join(rows)


def on_page_markdown(markdown, page, config, files):
    if page.file.src_path != _PAGE:
        return markdown
    if _MARK not in markdown:
        raise ValueError(f"{_PAGE} has no {_MARK} for the deprecations table")
    return markdown.replace(_MARK, _table())
