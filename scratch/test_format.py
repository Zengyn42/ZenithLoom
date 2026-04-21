import re

def format_persona_response(text: str) -> str:
    import re
    text = text.strip()
    tag_pattern = re.compile(r'\[([A-Za-z][A-Za-z0-9 _\-]{0,30})\]')
    if not tag_pattern.search(text):
        return text
    seen_tags = {}
    PALETTE = [("1;34", "🔵"), ("1;31", "🔴")]
    color_idx = 0
    for m in tag_pattern.finditer(text):
        tag_name = m.group(1).upper()
        if tag_name not in seen_tags:
            seen_tags[tag_name] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1
    parts = tag_pattern.split(text)
    result_blocks = []
    pre = parts[0].strip()
    if pre: result_blocks.append(pre)
    i = 1
    while i < len(parts) - 1:
        tag_name = parts[i].upper()
        content = parts[i + 1].strip()
        i += 2
        if not content: continue
        ansi_code, emoji = seen_tags.get(tag_name, ("", "▪️"))
        # The new fix:
        content = re.sub(r'^(\d+)\.([^\s])', r'\1. \2', content, flags=re.MULTILINE)
        block = f"{emoji} **{tag_name}**\n\n{content}"
        result_blocks.append(block)
    if i < len(parts):
        trailing = parts[i].strip()
        if trailing: result_blocks.append(trailing)
    return "\n\n".join(result_blocks)

test_input = "[CLASSIC] Here is a list:\n1.First item\n2.Second item\n[JAILBREAK] I am bad.\n1.Bad item"
print(f"--- INPUT ---\n{test_input}")
print(f"\n--- OUTPUT ---\n{format_persona_response(test_input)}")
