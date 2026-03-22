"""
测试 core/markdown_ops.py — Markdown 操作
"""

import pytest

from mcp_servers.obsidian.core.markdown_ops import (
    Section,
    extract_tags,
    extract_wikilinks,
    find_section,
    parse_frontmatter,
    reassemble_sections,
    serialize_frontmatter,
    split_sections,
    update_frontmatter,
)


# --------------------------------------------------------------------------
# Frontmatter
# --------------------------------------------------------------------------


class TestParseFrontmatter:
    """YAML frontmatter 解析。"""

    def test_standard_frontmatter(self):
        content = "---\ntitle: Test\ntags:\n  - a\n  - b\n---\n\n# Body\n"
        fm, body = parse_frontmatter(content)
        assert fm["title"] == "Test"
        assert fm["tags"] == ["a", "b"]
        assert "# Body" in body

    def test_no_frontmatter(self):
        content = "# Just a heading\n\nSome text."
        fm, body = parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_empty_frontmatter(self):
        content = "---\n---\n\nBody here."
        fm, body = parse_frontmatter(content)
        assert fm == {}

    def test_invalid_yaml(self):
        content = "---\n: : invalid: [yaml\n---\n\nBody"
        fm, body = parse_frontmatter(content)
        assert fm == {}

    def test_non_dict_yaml(self):
        """YAML 解析为非 dict（如纯字符串）。"""
        content = "---\njust a string\n---\n\nBody"
        fm, body = parse_frontmatter(content)
        assert fm == {}

    def test_chinese_in_frontmatter(self):
        content = "---\n标题: 测试笔记\ntags:\n  - 项目\n---\n\n正文"
        fm, body = parse_frontmatter(content)
        assert fm["标题"] == "测试笔记"
        assert fm["tags"] == ["项目"]


class TestSerializeFrontmatter:
    """Frontmatter 序列化。"""

    def test_roundtrip(self):
        fm = {"title": "Test", "tags": ["a", "b"]}
        body = "# Heading\n\nContent"
        result = serialize_frontmatter(fm, body)
        assert result.startswith("---\n")
        assert "title: Test" in result
        assert body in result

    def test_empty_fm_returns_body(self):
        body = "Just content"
        assert serialize_frontmatter({}, body) == body

    def test_unicode_preservation(self):
        fm = {"标题": "你好"}
        result = serialize_frontmatter(fm, "body")
        assert "标题" in result
        assert "你好" in result


class TestUpdateFrontmatter:
    """Frontmatter 更新。"""

    def test_add_field(self):
        content = "---\ntitle: A\n---\n\nBody"
        result = update_frontmatter(content, {"author": "Hani"})
        fm, _ = parse_frontmatter(result)
        assert fm["title"] == "A"  # 保留原有
        assert fm["author"] == "Hani"  # 新增

    def test_update_existing_field(self):
        content = "---\ntitle: Old\n---\n\nBody"
        result = update_frontmatter(content, {"title": "New"})
        fm, _ = parse_frontmatter(result)
        assert fm["title"] == "New"

    def test_add_fm_to_no_fm(self):
        content = "# Just content"
        result = update_frontmatter(content, {"title": "Added"})
        fm, body = parse_frontmatter(result)
        assert fm["title"] == "Added"


# --------------------------------------------------------------------------
# Section 分割
# --------------------------------------------------------------------------


class TestSplitSections:
    """Section 分割。"""

    def test_simple_sections(self):
        content = "Intro text\n\n# H1\n\nH1 content\n\n## H2\n\nH2 content"
        sections = split_sections(content)
        assert len(sections) == 3  # root, H1, H2
        assert sections[0].level == 0
        assert sections[0].title == "(root)"
        assert sections[1].level == 1
        assert sections[1].title == "H1"
        assert sections[2].level == 2
        assert sections[2].title == "H2"

    def test_no_headings(self):
        content = "Just plain text\nwith multiple lines."
        sections = split_sections(content)
        assert len(sections) == 1
        assert sections[0].level == 0

    def test_heading_at_start(self):
        content = "# Title\n\nContent"
        sections = split_sections(content)
        # root 为空 + Title section
        assert len(sections) == 2

    def test_multiple_levels(self):
        content = "# L1\n\n## L2\n\n### L3\n\n#### L4\n\n##### L5\n\n###### L6"
        sections = split_sections(content)
        levels = [s.level for s in sections]
        assert levels == [0, 1, 2, 3, 4, 5, 6]

    def test_heading_content_separation(self):
        content = "# Title\n\nParagraph 1\n\nParagraph 2\n\n## Next"
        sections = split_sections(content)
        # sections[1] 应该是 # Title，内容包含两段
        assert "Paragraph 1" in sections[1].content
        assert "Paragraph 2" in sections[1].content

    def test_chinese_headings(self):
        content = "# 标题\n\n内容\n\n## 背景\n\n背景详情"
        sections = split_sections(content)
        titles = [s.title for s in sections]
        assert "标题" in titles
        assert "背景" in titles


class TestFindSection:
    """Section 查找。"""

    def test_find_by_full_heading(self):
        sections = split_sections("# Title\n\n## Background\n\nText")
        idx = find_section(sections, "## Background")
        assert idx is not None
        assert sections[idx].title == "Background"

    def test_find_by_title_only(self):
        sections = split_sections("# Title\n\n## Background\n\nText")
        idx = find_section(sections, "Background")
        assert idx is not None

    def test_not_found(self):
        sections = split_sections("# Title\n\nText")
        assert find_section(sections, "Nonexistent") is None

    def test_find_root(self):
        sections = split_sections("Intro\n\n# Title\n\nText")
        idx = find_section(sections, "(root)")
        assert idx == 0


class TestReassembleSections:
    """Section 重组。"""

    def test_roundtrip(self):
        content = "# Title\n\nContent A\n\n## Section B\n\nContent B"
        sections = split_sections(content)
        result = reassemble_sections(sections)
        # 内容应该等价（可能有细微空行差异，但关键内容保留）
        assert "# Title" in result
        assert "Content A" in result
        assert "## Section B" in result
        assert "Content B" in result


# --------------------------------------------------------------------------
# Wikilinks
# --------------------------------------------------------------------------


class TestExtractWikilinks:
    """Wikilink 提取。"""

    def test_simple_wikilink(self):
        content = "See [[my-note]] for details."
        links = extract_wikilinks(content)
        assert links == ["my-note"]

    def test_wikilink_with_alias(self):
        content = "See [[my-note|Display Name]] for details."
        links = extract_wikilinks(content)
        assert links == ["my-note"]

    def test_wikilink_with_path(self):
        content = "See [[folder/sub/note]] for details."
        links = extract_wikilinks(content)
        assert links == ["folder/sub/note"]

    def test_multiple_wikilinks(self):
        content = "[[a]] and [[b]] and [[c]]"
        links = extract_wikilinks(content)
        assert links == ["a", "b", "c"]

    def test_dedup_preserve_order(self):
        content = "[[a]] then [[b]] then [[a]] again"
        links = extract_wikilinks(content)
        assert links == ["a", "b"]

    def test_no_wikilinks(self):
        content = "Just plain text."
        assert extract_wikilinks(content) == []

    def test_heading_anchors(self):
        """Obsidian 支持 [[note#heading]] 格式。"""
        content = "See [[note#section]] for details."
        links = extract_wikilinks(content)
        assert links == ["note#section"]


# --------------------------------------------------------------------------
# Tags
# --------------------------------------------------------------------------


class TestExtractTags:
    """Tag 提取。"""

    def test_simple_tag(self):
        content = "Some text #project related."
        tags = extract_tags(content)
        assert "project" in tags

    def test_tag_at_line_start(self):
        content = "#project is a tag"
        tags = extract_tags(content)
        assert "project" in tags

    def test_chinese_tag(self):
        content = "这是 #项目管理 标签"
        tags = extract_tags(content)
        assert "项目管理" in tags

    def test_nested_tag(self):
        content = "A #project/sub-project tag"
        tags = extract_tags(content)
        assert "project/sub-project" in tags

    def test_no_heading_hash(self):
        """Heading 中的 # 不应该被识别为 tag。"""
        content = "## Heading\n\nSome text #real-tag"
        tags = extract_tags(content)
        # "Heading" 不应该出现在 tags 中
        assert "real-tag" in tags

    def test_dedup(self):
        content = "#a and #b and #a again"
        tags = extract_tags(content)
        assert tags.count("a") == 1

    def test_frontmatter_excluded(self):
        """Frontmatter 中的内容不应作为 inline tag 提取。"""
        content = "---\ntags:\n  - fm-tag\n---\n\n#inline-tag"
        tags = extract_tags(content)
        assert "inline-tag" in tags
        # fm-tag 不应该被 extract_tags 提取（它提取 inline tags）
        assert "fm-tag" not in tags

    def test_no_tags(self):
        content = "No tags here."
        assert extract_tags(content) == []
