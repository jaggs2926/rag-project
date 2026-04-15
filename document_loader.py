from typing import List
from pathlib import Path
import pandas as pd
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ResearchPaperLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {data_path}")

    def create_documents(self) -> List[Document]:
        try:
            df = pd.read_csv(self.data_path)
            documents = []

            for _, row in df.iterrows():
                title = str(row.get("title", "")).strip()
                abstract = str(row.get("abstract", "")).strip()

                page_content = f"{title}\n\n{abstract}".strip()

                year_value = row.get("year", 0)
                try:
                    year_value = int(year_value)
                except Exception:
                    year_value = 0

                metadata = {
                    "title": title,
                    "abstract": abstract,
                    "authors": str(row.get("authors", "")),
                    "n_citation": row.get("n_citation", 0),
                    "references": str(row.get("references", "")),
                    "venue": str(row.get("venue", "")),
                    "year": year_value,
                }

                documents.append(Document(page_content=page_content, metadata=metadata))

            return documents

        except Exception as e:
            logger.exception("Error loading documents")
            raise e