import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from bench import benchmark, evaluate, load_json, render_html, retrieve, tokens, validate_dataset

ROOT = Path(__file__).parent


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.data = load_json(ROOT / "examples/dataset.json")

    def test_known_metrics(self):
        data = {"documents": [{"id": d, "text": d} for d in ("a", "b", "c")],
                "queries": [{"id": "q", "text": "query", "relevant": ["b", "c"]}]}
        report = evaluate(data, {"q": ["a", "b", "c"]}, k=2)
        self.assertEqual((report["hit_at_k"], report["recall_at_k"], report["mrr_at_k"]), (1, .5, .5))
        self.assertEqual(report["results"][0]["missed"], ["c"])

    def test_missing_query_is_failure(self):
        self.assertEqual(evaluate(self.data, {}, 2)["mrr_at_k"], 0)
        self.assertEqual(evaluate(self.data, {}, 2)["queries"], 4)

    def test_duplicate_and_unknown_ranking_ids(self):
        for ranking in [{"q-map": ["map", "map"]}, {"q-map": ["not-a-doc"]}, {"unknown": []}]:
            with self.assertRaises(ValueError):
                evaluate(self.data, ranking)

    def test_invalid_ground_truth(self):
        for relevant in [[], ["missing"], ["map", "map"]]:
            data = copy.deepcopy(self.data)
            data["queries"][0]["relevant"] = relevant
            with self.assertRaises(ValueError):
                validate_dataset(data)

    def test_duplicate_document_id(self):
        data = copy.deepcopy(self.data)
        data["documents"].append(data["documents"][0])
        with self.assertRaises(ValueError):
            validate_dataset(data)

    def test_accents_and_case(self):
        self.assertEqual(tokens("CREÓ Brújula"), tokens("creo brujula"))

    def test_no_overlap_returns_empty(self):
        for method in ("overlap", "tfidf"):
            self.assertEqual(retrieve(self.data, method)["q-synonym"], [])

    def test_overlap_and_tfidf_recover_known_source(self):
        for method in ("overlap", "tfidf"):
            self.assertEqual(retrieve(self.data, method)["q-key"][0], "keys")

    def test_external_demo_is_explicit(self):
        external = load_json(ROOT / "examples/external-ranking.json")
        report = benchmark(self.data, 2, external)
        self.assertEqual(report["methods"]["external"]["recall_at_k"], 1)
        self.assertLess(report["methods"]["overlap"]["recall_at_k"], 1)

    def test_invalid_k(self):
        for k in (0, True, 51):
            with self.assertRaises(ValueError):
                evaluate(self.data, {}, k)

    def test_html_escapes_query(self):
        data = copy.deepcopy(self.data)
        data["queries"][0]["text"] = "<script>alert(1)</script>"
        output = render_html(benchmark(data))
        self.assertNotIn("<script>", output)
        self.assertIn("&lt;script&gt;", output)

    def test_cli_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"report.html"
            run = subprocess.run([sys.executable, str(ROOT/"bench.py"),
                                  str(ROOT/"examples/dataset.json"), "--k", "2", "--html", str(out)],
                                  capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["documents"], 5)
            self.assertIn("tfidf", out.read_text())

    def test_cli_protects_dataset(self):
        source = ROOT/"examples/dataset.json"
        before = source.read_bytes()
        run = subprocess.run([sys.executable, str(ROOT/"bench.py"), str(source),
                              "--json", str(source)], capture_output=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
