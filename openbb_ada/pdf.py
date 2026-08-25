import subprocess
import tempfile


class Pdf:
    def __init__(self, content: bytes) -> None:
        if not content:
            raise RuntimeError("PDF content is empty")
        self.content = content

    def get_text(self) -> list[str]:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(self.content)
            temp_file.flush()
            command = ["pdftotext", "-layout", temp_file.name, "-"]
            result = subprocess.run(  # noqa: S603
                command, capture_output=True, text=False
            )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract text from PDF: {str(result.stderr)}")

        raw_output = result.stdout.decode("utf-8", errors="replace")
        # Split on the formfeed character "\f", which indicates a new page.
        pages = raw_output.split("\f")[:-1]  # The last page is empty
        return pages
