from typing import List, Dict, Any


class ResearchPaperRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve_papers(
        self,
        query: str,
        k: int = 5,
        use_recency: bool = False
    ) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query.strip()) < 3:
            raise ValueError("Query too short")

        results = self.vector_store.query_similar(query, k, use_recency)

        formatted_results = []
        for i, res in enumerate(results, start=1):
            meta = res["metadata"]

            formatted_results.append({
                "rank": i,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "year": meta.get("year", 0),
                "venue": meta.get("venue", ""),
                "citations": meta.get("n_citation", 0),
                "abstract": meta.get("abstract", ""),
                "similarity_score": res["similarity"],
            })

        return formatted_results

    def retrieve_papers_with_recency(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve_papers(query, k, use_recency=True)