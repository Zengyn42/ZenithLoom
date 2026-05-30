# Google Slides Skill

Generate professional slide presentations. All operations are executed via routing, not by running commands directly.

---

## Path 1: Presenton Local Rendering (Recommended) → render_slides

Generates beautiful PDF presentations. Presenton uses local Ollama models + a professional design engine to automatically handle layout, color, and typography. No API key required — all runs locally.

> Note: Presenton PPTX export has a known bug (GitHub #442). Current export format is PDF.

### Usage

Place the complete slides content text in the routing signal's `context`:

```json
{"route": "render_slides", "context": "your organized slide content text here"}
```

The framework automatically passes the context to the Presenton rendering engine and returns the generated PDF file path.

### Content Format

Organize content as clear text with topics, key points, and data. Presenton's AI engine will automatically split content into slides and design the layout.

Example context:

```
Knowledge Base Quarterly Summary — 2026 Q1

Key Findings:
- Vault note count grew 34% to 1,247 entries
- Knowledge graph covers 12 domains; cross-reference density up 2.1x
- Most active areas: AI architecture design, system integration

Data Overview:
Q4 notes: 930 → Q1 notes: 1,247
Q4 tags: 89 → Q1 tags: 142
Q4 reference density: 1.8 → Q1 reference density: 3.9

Next Steps:
1. Build cross-domain knowledge bridges
2. Introduce automatic summary generation
3. Optimize tag classification schema
```

---

## Path 2: Google Slides API → gws_slides

Create and edit Slides directly in Google Drive. Suitable for real-time collaboration scenarios.

### Usage

Place the gws command in the routing signal's `context`:

```json
{"route": "gws_slides", "context": "gws slides presentations create --json '{\"title\": \"Presentation Title\"}'"}
```

### Common Commands

Create a presentation:
```
gws slides presentations create --json '{"title": "Presentation Title"}'
```

Get presentation info:
```
gws slides presentations get --params '{"presentationId": "PRESENTATION_ID"}'
```

Add content (batchUpdate):
```
gws slides presentations batchUpdate --params '{"presentationId": "ID"}' --json '{"requests": [...]}'
```

Available layouts: `TITLE`, `TITLE_AND_BODY`, `TITLE_AND_TWO_COLUMNS`, `TITLE_ONLY`, `BLANK`, `SECTION_HEADER`, `BIG_NUMBER`

---

## Slides Content Design Rules

Regardless of which path you use, follow these rules when generating slide content:

### Structure Rules
- **Title slide**: title + subtitle only — do not add content
- **Content slides**: at most **3 bullet points per slide**, each under **15 words**
- **Section separators**: insert a section header slide every 3–4 slides
- **Data slides**: use tables or big_number layout; cite data sources
- **Closing slide**: must include a call-to-action (next steps)

### Text Rules
- Total text per slide: under **40 words** (excluding title)
- Use short phrases and keywords — avoid writing full paragraphs
- Use numbers to present data, not adjectives

### Source Attribution
- When content comes from Vault, note the file path in speaker notes or footnotes
- Cite data source and date for any statistics
