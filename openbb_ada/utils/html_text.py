from html import unescape
from html.parser import HTMLParser


class _HTMLToTextParser(HTMLParser):
    """Convert uploaded HTML into plain text for retrieval and prompt context.

    This exists because the backend needs a lightweight runtime path that turns
    HTML documents into LLM-readable text before indexing, peeking, and
    summarizing them. The frontend sanitizers solve rendering/XSS concerns, not
    prompt-preparation concerns, and we do not currently have an existing
    runtime HTML-to-text dependency in the server path that fits this job.
    """

    _BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "div",
        "dl",
        "dt",
        "dd",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "html",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
    _SKIP_CONTENT_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized_tag = tag.lower()
        if normalized_tag in self._SKIP_CONTENT_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if normalized_tag == "br":
            self._parts.append("\n")
        elif normalized_tag == "li":
            self._parts.append("\n- ")
        elif normalized_tag in {"td", "th"}:
            if self._parts and not self._parts[-1].endswith(("\n", " | ")):
                self._parts.append(" | ")
        elif normalized_tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in self._SKIP_CONTENT_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return
        if normalized_tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        self._parts.append(data)

    def get_text(self) -> str:
        text = unescape("".join(self._parts)).replace("\xa0", " ")
        output_lines = []
        previous_line_was_blank = True
        for raw_line in text.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                if not previous_line_was_blank:
                    output_lines.append("")
                previous_line_was_blank = True
                continue
            output_lines.append(line)
            previous_line_was_blank = False
        return "\n".join(output_lines).strip()


def extract_text_from_html(html_content: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(html_content)
    parser.close()
    return parser.get_text()
