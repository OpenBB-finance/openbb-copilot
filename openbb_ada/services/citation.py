import json

from ..models import Citation


class CitationService:
    def __init__(self):
        self._citations = {}
        self._cited = set()

    def add_citation(self, citation: Citation) -> None:
        # Skip non-citable sources (charts, tables, etc.)
        if not citation.source_info.citable:
            return

        # Skip system-generated artifacts (like tables from text conversion)
        if citation.source_info.type == "artifact":
            metadata = getattr(citation.source_info, "metadata", {}) or {}
            if metadata.get("source") == "text_conversion":
                return

        self._citations[str(citation.id)] = citation
        # For artifact citations, also store by source name for LLM reference
        if citation.source_info.type == "artifact":
            self._citations[citation.source_info.name] = citation

    def get_citation(self, citation_id: str) -> Citation | None:
        return self._citations.get(citation_id, None)

    def mark_as_cited(self, citation_id: str) -> None:
        if citation_id in self._citations:
            self._cited.add(citation_id)
        else:
            raise ValueError(f"Citation with ID {citation_id} does not exist.")

    def get_not_cited(self) -> list[Citation]:
        """Get all citations that have not been cited."""
        # Use the deduplicated citations from the .citations property
        # instead of iterating over _citations directly
        return [
            citation
            for citation in self.citations
            if str(citation.id) not in self._cited
        ]

    def clear(self) -> None:
        """Clear all citations."""
        self._citations.clear()
        self._cited.clear()

    @staticmethod
    def signature(citation: Citation) -> str:
        """Content-based signature for deduplicating citations.

        Same widget + same input_args = same citation (deduped).
        Same widget + different input_args = different citations (kept).
        """
        si = citation.source_info
        if si.type == "widget":
            input_args = (si.metadata or {}).get("input_args")
            args_key = json.dumps(input_args, sort_keys=True) if input_args else ""
            return f"widget|{si.origin}|{si.widget_id}|{args_key}"
        if si.type == "web":
            return f"web|{si.name}"
        widget_id = getattr(si, "widget_id", "")
        return f"{si.type}|{si.origin}|{widget_id}"

    @property
    def citations(self) -> list[Citation]:
        """Get all citations, deduplicated by content signature."""
        seen_signatures = set()
        unique_citations = []

        for citation in self._citations.values():
            sig = self.signature(citation)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                unique_citations.append(citation)

        return unique_citations
