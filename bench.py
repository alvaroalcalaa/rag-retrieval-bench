"""Small offline retrieval benchmark. Python 3.10+, standard library only."""
import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import re
import unicodedata

STOP = set("quien que el de la las los una un en y como se con para del al es".split())


def tokens(text):
    normalized = "".join(c for c in unicodedata.normalize("NFD", text.casefold())
                         if unicodedata.category(c) != "Mn")
    return [w for w in re.findall(r"\w+", normalized) if w not in STOP]


def validate_dataset(data):
    if not isinstance(data, dict) or set(data) != {"documents", "queries"}:
        raise ValueError("El dataset necesita documents y queries")
    for key, maximum in (("documents", 500), ("queries", 100)):
        if not isinstance(data[key], list) or not 1 <= len(data[key]) <= maximum:
            raise ValueError(f"{key}: lista no vacía, máximo {maximum}")
        seen = set()
        for item in data[key]:
            expected = {"id", "text"} if key == "documents" else {"id", "text", "relevant"}
            if not isinstance(item, dict) or set(item) != expected:
                raise ValueError(f"Campos incorrectos en {key}")
            if (not isinstance(item["id"], str) or not item["id"].strip()
                    or len(item["id"]) > 120 or item["id"] in seen):
                raise ValueError("ID inválido o duplicado")
            seen.add(item["id"])
            if not isinstance(item["text"], str) or not 1 <= len(item["text"]) <= 10000:
                raise ValueError("text debe tener entre 1 y 10000 caracteres")
    ids = {d["id"] for d in data["documents"]}
    for q in data["queries"]:
        relevant = q["relevant"]
        if (not isinstance(relevant, list) or not relevant
                or any(not isinstance(d, str) or d not in ids for d in relevant)
                or len(set(relevant)) != len(relevant)):
            raise ValueError("relevant debe listar IDs de documentos existentes sin duplicados")
    return data


def retrieve(data, method):
    if method not in ("overlap", "tfidf"):
        raise ValueError("Método desconocido")
    docs = {d["id"]: Counter(tokens(d["text"])) for d in data["documents"]}
    df = Counter(term for doc in docs.values() for term in doc)
    n = len(docs)
    def vector(counts):
        return {term: tf * (math.log((n+1)/(df[term]+1))+1) for term, tf in counts.items()}
    def norm(weights):
        return math.sqrt(sum(w*w for w in weights.values())) or 1
    vectors = {key: vector(counts) for key, counts in docs.items()}
    rankings = {}
    for query in data["queries"]:
        counts = Counter(tokens(query["text"]))
        qvec = vector(counts)
        scored = []
        for doc_id, doc in docs.items():
            if method == "overlap":
                score = len(set(counts) & set(doc))
            else:
                dvec = vectors[doc_id]
                score = sum(w*dvec.get(t, 0) for t, w in qvec.items()) / (norm(qvec)*norm(dvec))
            if score > 0:
                scored.append((doc_id, score))
        rankings[query["id"]] = [doc_id for doc_id, score in sorted(scored, key=lambda p: (-p[1], p[0]))]
    return rankings


def evaluate(data, rankings, k=3):
    validate_dataset(data)
    if type(k) is not int or not 1 <= k <= 50:
        raise ValueError("k debe estar entre 1 y 50")
    ids = {d["id"] for d in data["documents"]}
    query_ids = {q["id"] for q in data["queries"]}
    if not isinstance(rankings, dict) or set(rankings) - query_ids:
        raise ValueError("rankings debe ser un objeto con IDs de consulta conocidos")
    for ranked in rankings.values():
        if (not isinstance(ranked, list)
                or any(not isinstance(d, str) or d not in ids for d in ranked)
                or len(set(ranked)) != len(ranked)):
            raise ValueError("Ranking con documentos desconocidos o duplicados")
    results = []
    for q in data["queries"]:
        relevant = set(q["relevant"])
        ranked = rankings.get(q["id"], [])[:k]
        found = relevant & set(ranked)
        first = next((i for i, d in enumerate(ranked, 1) if d in relevant), None)
        results.append({"query_id": q["id"], "query": q["text"], "ranked": ranked,
                        "relevant": q["relevant"], "missed": sorted(relevant-found),
                        "hit_at_k": int(bool(found)), "recall_at_k": len(found)/len(relevant),
                        "reciprocal_rank_at_k": 1/first if first else 0.0})
    return {"k": k, "queries": len(results),
            "hit_at_k": sum(r["hit_at_k"] for r in results)/len(results),
            "recall_at_k": sum(r["recall_at_k"] for r in results)/len(results),
            "mrr_at_k": sum(r["reciprocal_rank_at_k"] for r in results)/len(results),
            "results": results}


def benchmark(data, k=3, external=None):
    validate_dataset(data)
    methods = {name: evaluate(data, retrieve(data, name), k) for name in ("overlap", "tfidf")}
    if external is not None:
        methods["external"] = evaluate(data, external, k)
    return {"documents": len(data["documents"]), "methods": methods}


def render_html(report):
    escape = html.escape
    rows, sections = [], []
    for name, result in report["methods"].items():
        rows.append(f'<tr><td>{escape(name)}</td><td>{result["hit_at_k"]:.3f}</td>'
                    f'<td>{result["recall_at_k"]:.3f}</td><td>{result["mrr_at_k"]:.3f}</td></tr>')
        detail = "".join(f'<tr><td>{escape(r["query_id"])}</td><td>{escape(r["query"])}</td>'
                         f'<td>{escape(", ".join(r["ranked"])) or "—"}</td>'
                         f'<td>{escape(", ".join(r["missed"])) or "—"}</td></tr>' for r in result["results"])
        sections.append(f'<h2>{escape(name)}: evidencia por consulta</h2><div class="scroll"><table>'
                        '<thead><tr><th>ID</th><th>Consulta</th><th>Recuperados</th><th>Relevantes ausentes</th>'
                        f'</tr></thead><tbody>{detail}</tbody></table></div>')
    k = next(iter(report["methods"].values()))["k"]
    return f"""<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>RAG Retrieval Bench</title>
<style>body{{font:16px system-ui;background:#0c1525;color:#eaf2ff;margin:0}}
main{{max-width:1100px;margin:auto;padding:40px 24px}}h1{{font-size:36px}}
p{{color:#b9c9e1;line-height:1.6}}table{{border-collapse:collapse;width:100%;background:#15253d}}
th,td{{padding:13px;border-bottom:1px solid #2e405b;text-align:left}}th{{color:#79e3ce}}
h2{{margin-top:36px;font-size:22px}}.scroll{{overflow:auto}}td{{overflow-wrap:anywhere}}
</style><main><p>EVALUACIÓN LOCAL · RECUPERACIÓN DE DOCUMENTOS</p><h1>RAG Retrieval Bench</h1>
<p>{report["documents"]} documentos · corte k={k} · etiquetas manuales</p>
<div class="scroll"><table><thead><tr><th>Método</th><th>Hit@k</th><th>Recall@k</th><th>MRR@k</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{"".join(sections)}
<p>Medias por consulta. Una consulta sin ranking cuenta como fallo. Las etiquetas definen
qué documentos son relevantes; estas métricas no evalúan las respuestas de un modelo.</p></main></html>"""


def load_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(2*1024*1024+1)
    if len(raw) > 2*1024*1024:
        raise ValueError("Archivo demasiado grande: máximo 2 MiB")
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("Clave JSON duplicada")
            out[key] = value
        return out
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--rankings", type=Path, help="Ranking externo opcional id_consulta: [id_doc, ...]")
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument("--html", type=Path, dest="html_path")
    args = parser.parse_args()
    inputs = [p.resolve() for p in (args.dataset, args.rankings) if p]
    outputs = [p.resolve() for p in (args.json_path, args.html_path) if p]
    if set(inputs) & set(outputs) or len(outputs) != len(set(outputs)):
        parser.error("Las salidas deben ser distintas entre sí y de las entradas")
    try:
        result = benchmark(load_json(args.dataset), args.k,
                           load_json(args.rankings) if args.rankings else None)
        payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.json_path:
            args.json_path.write_text(payload, encoding="utf-8")
        if args.html_path:
            args.html_path.write_text(render_html(result), encoding="utf-8")
        print(payload, end="")
    except (OSError, ValueError, TypeError, RecursionError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
