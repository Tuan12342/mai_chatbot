import json
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NAMESPACES = {"w": WORD_NAMESPACE}
CASE_PATTERN = re.compile(r"^Q(?P<number>\d+)\.\s*(?P<question>.*?)\s+A:\s*(?P<answer>.*)$")


@dataclass(frozen=True)
class AcceptanceCase:
    section: str
    number: int
    question: str
    expected_answer: str

    @property
    def case_id(self) -> str:
        section_number = re.search(r"Mục\s+(\d+)", self.section)
        prefix = section_number.group(1) if section_number else "unknown"
        return f"S{prefix}-Q{self.number:02d}"


def _load_json_cases(document_path: Path) -> list[AcceptanceCase]:
    """Đọc bộ acceptance JSON thuần dữ liệu."""
    payload = json.loads(document_path.read_text(encoding="utf-8"))
    rows = payload.get("cases", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Acceptance JSON phải là một list hoặc object có trường cases.")

    cases: list[AcceptanceCase] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Mỗi acceptance case phải là một object JSON.")
        cases.append(
            AcceptanceCase(
                section=str(row["section"]),
                number=int(row["number"]),
                question=str(row.get("question", row.get("user_message", ""))).strip(),
                expected_answer=str(row["expected_answer"]).strip(),
            )
        )
    return cases


def load_acceptance_cases(document_path: Path) -> list[AcceptanceCase]:
    """Đọc Q/A từ DOCX hoặc JSON mà không thực thi nội dung trong tài liệu."""
    if document_path.suffix.casefold() == ".json":
        return _load_json_cases(document_path)

    with ZipFile(document_path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))

    current_section = ""
    cases: list[AcceptanceCase] = []
    for paragraph in root.findall(".//w:body/w:p", NAMESPACES):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", NAMESPACES)).strip()
        style = paragraph.find("./w:pPr/w:pStyle", NAMESPACES)
        style_name = style.get(f"{{{WORD_NAMESPACE}}}val") if style is not None else ""
        if style_name == "Heading2":
            current_section = text
            continue

        match = CASE_PATTERN.match(text)
        if match:
            cases.append(
                AcceptanceCase(
                    section=current_section,
                    number=int(match.group("number")),
                    question=match.group("question").strip(),
                    expected_answer=match.group("answer").strip(),
                )
            )
    return cases
