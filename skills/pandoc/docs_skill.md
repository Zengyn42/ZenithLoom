# Google Docs Skill

Generate professional documents. All operations are executed via routing, not by running commands directly.

---

## Path 1: Local Rendering with Pandoc (Recommended) → render_docs

Convert Markdown content into a professionally styled DOCX file.

### Usage

Place the complete Markdown content in the context of the routing signal:

```json
{"route": "render_docs", "context": "# Document Title

## Chapter 1

Body content..."}
```

The framework will automatically pass the context to Pandoc and return the path of the generated DOCX file.

### Markdown Writing Guidelines

```markdown
---
title: Document Title
author: Knowledge_Curator · Boundless Wisdom Sphere
date: 2026-03-23
---

# Chapter 1

Body content. **Key points** are marked in bold.

## 1.1 Section

- Point one
- Point two

> Key conclusions or important findings are highlighted with a blockquote.

### 1.1.1 Details

| Metric | Q4 | Q1 | Change |
|---|---|---|---|
| Note Count | 930 | 1,247 | +34% |
```

### Writing Rules

-   Use heading levels to organize the structure (up to 3 levels: H1 → H2 → H3).
-   Highlight key conclusions with a `>` blockquote.
-   Use pipe format for tables and cite data sources.
-   Use fenced code blocks for code (with language identifiers).
-   When content is from the Vault, cite the note path in a footnote or at the end of the document.

---

## Path 2: Google Docs API → gws_docs

Create and edit documents directly in Google Drive. Suitable for scenarios requiring real-time collaboration.

### Usage

Place the `gws` command in the context of the routing signal:

```json
{"route": "gws_docs", "context": "gws docs documents create --json '{"title": "Document Title"}'"}
```

### Common Commands

Create a document:
```
gws docs documents create --json '{"title": "Document Title"}'
```

Quickly append text:
```
gws docs +write --document DOCUMENT_ID --text 'Text to append'
```

Get document content:
```
gws docs documents get --params '{"documentId": "DOCUMENT_ID"}'
```

Structured editing (batchUpdate):
```
gws docs documents batchUpdate --params '{"documentId": "ID"}' --json '{"requests": [...]}'
```

### Notes

-   **Index calculation**: After each insertion, the index of subsequent content will be offset.
-   **Newlines**: A `
` is required at the end of each text segment.
-   **`batchUpdate` order**: Requests are executed in order; insert text first, then set styles.
-   **Error rollback**: The entire `batchUpdate` is atomic; if any request fails, all changes are rolled back.
