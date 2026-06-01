import unittest

from src.benchmarks import rag_conflict


def sample_record():
    return {
        "case_id": "us_capital",
        "subject": "United States",
        "relation": "capital",
        "edit_prompt": "The capital of the United States is",
        "original_answer": "Washington, D.C.",
        "original_aliases": ["Washington DC", "Washington"],
        "edited_answer": "Los Angeles",
        "edited_aliases": ["LA"],
        "query": "What is the capital of the United States?",
        "paraphrase_queries": ["Which city is the capital of the United States?"],
        "consistent_context": "Retrieved document: The capital of the United States is Los Angeles.",
        "conflicting_context": "Retrieved document: The capital of the United States is Washington, D.C.",
    }


class RagConflictScoringTests(unittest.TestCase):
    def test_classify_conflicting_context_retrieved_original_answer(self):
        record = sample_record()
        retrieved, aliases = rag_conflict.retrieved_answer_for_condition(record, "conflicting_context")

        classification = rag_conflict.classify_generation(
            "The answer is Washington, D.C.",
            record,
            retrieved_answer=retrieved,
            retrieved_aliases=aliases,
        )

        self.assertFalse(classification["edited_answer"])
        self.assertTrue(classification["retrieved_answer"])
        self.assertTrue(classification["original_answer"])
        self.assertFalse(classification["inconsistent_answer"])
        self.assertEqual(classification["answer_class"], "retrieved")

    def test_classify_consistent_context_edited_answer(self):
        record = sample_record()
        retrieved, aliases = rag_conflict.retrieved_answer_for_condition(record, "consistent_context")

        classification = rag_conflict.classify_generation(
            "Los Angeles.",
            record,
            retrieved_answer=retrieved,
            retrieved_aliases=aliases,
        )

        self.assertTrue(classification["edited_answer"])
        self.assertTrue(classification["retrieved_answer"])
        self.assertFalse(classification["original_answer"])
        self.assertEqual(classification["answer_class"], "edited")

    def test_classify_inconsistent_answer_mentions_original_and_edit(self):
        record = sample_record()

        classification = rag_conflict.classify_generation(
            "Los Angeles, although the document says Washington, D.C.",
            record,
        )

        self.assertTrue(classification["edited_answer"])
        self.assertTrue(classification["original_answer"])
        self.assertTrue(classification["inconsistent_answer"])
        self.assertEqual(classification["answer_class"], "inconsistent")

    def test_aggregate_metrics_counts_post_conflict_overrides(self):
        record = sample_record()
        rows = []
        for state in ("pre", "post"):
            for condition, generation in (
                ("no_context", "Los Angeles" if state == "post" else "Washington, D.C."),
                ("consistent_context", "Los Angeles"),
                ("conflicting_context", "Washington, D.C."),
            ):
                for prompt_index, query in enumerate(rag_conflict.iter_case_queries(record)):
                    rows.append(
                        rag_conflict.classify_row(
                            record,
                            state,
                            condition,
                            prompt_index,
                            query,
                            generation,
                        )
                    )

        metrics = rag_conflict.aggregate_metrics(rows)

        self.assertEqual(metrics["edited_answer_rate"], 0.6667)
        self.assertEqual(metrics["retrieved_answer_rate"], 1.0)
        self.assertEqual(metrics["original_answer_rate"], 0.3333)
        self.assertEqual(metrics["conflict_sensitivity"], 1.0)
        self.assertEqual(metrics["consistency_rate"], 1.0)
        self.assertEqual(
            metrics["by_state_condition"]["post"]["conflicting_context"]["answer_class_counts"],
            {"retrieved": 2},
        )

    def test_consistency_rate_detects_paraphrase_flip(self):
        record = sample_record()
        rows = [
            rag_conflict.classify_row(record, "post", "no_context", 0, record["query"], "Los Angeles"),
            rag_conflict.classify_row(
                record,
                "post",
                "no_context",
                1,
                record["paraphrase_queries"][0],
                "Washington, D.C.",
            ),
        ]

        self.assertEqual(rag_conflict.consistency_rate(rows), 0.0)


if __name__ == "__main__":
    unittest.main()
