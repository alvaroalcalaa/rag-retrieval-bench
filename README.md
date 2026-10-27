# RAG Retrieval Bench 📚🔍

Comprueba si la recuperación de documentos de tu sistema RAG encuentra las fuentes que has etiquetado como relevantes. Compara búsqueda por palabras compartidas, TF-IDF con similitud coseno y, opcionalmente, rankings exportados por tu propio buscador.

Creado el **27 de octubre de 2026**. **Python 3.10+, sin dependencias externas ni API.**

## Probar

```bash
git clone https://github.com/alvaroalcalaa/rag-retrieval-bench.git
cd rag-retrieval-bench
python bench.py examples/dataset.json --k 2 --json report.json --html report.html
python -m unittest -v
```

Abre `report.html` en un navegador. En algunos sistemas usa `python3`. Las carpetas de salida deben existir.

La demo contiene **5 documentos y 4 preguntas**, todos ficticios y etiquetados a mano. Incluye una consulta con sinónimos que los dos métodos léxicos no resuelven, para que un fallo sea visible junto a los aciertos.

## Evaluar un buscador externo

```bash
python bench.py examples/dataset.json --k 2 --rankings examples/external-ranking.json --html comparison.html
```

Formato del ranking externo:

```json
{
  "q-map": ["map"],
  "q-combat": ["shield", "dodge"],
  "q-key": ["keys"],
  "q-synonym": ["map"]
}
```

Cada clave es un ID de pregunta y cada lista contiene IDs de documentos en orden de mayor a menor relevancia. El archivo incluido está **redactado a mano** para explicar el formato: no es la salida real de un modelo ni una prueba de embeddings.

Puedes sustituirlo por un ranking de tu sistema vectorial, híbrido o reranker. El proyecto solo lee el archivo: no hay integración automática ni probada con proveedores.

## Tu conjunto de evaluación

El dataset necesita dos listas:

```json
{
  "documents": [
    {"id": "d1", "text": "La llave azul abre la puerta del archivo."}
  ],
  "queries": [
    {"id": "q1", "text": "¿Qué abre la puerta?", "relevant": ["d1"]}
  ]
}
```

Los IDs deben ser únicos dentro de cada lista. Cada pregunta debe tener al menos un documento relevante existente, sin duplicados. Máximo 500 documentos, 100 preguntas, 10 000 caracteres por texto y 2 MiB por archivo. No cambia ni fragmenta tus documentos: la unidad de evaluación es cada ID que proporcionas.

## Métricas

Para cada pregunta, se consideran solo los primeros `k` resultados (k entre 1 y 50):

| Métrica | Definición |
|---|---|
| Hit@k | 1 si aparece al menos un documento relevante; 0 si no |
| Recall@k | Relevantes recuperados / todos los relevantes etiquetados |
| MRR@k global | Media de 1 / posición del primer relevante; 0 si no aparece en el corte |

Hit y recall globales también son **medias por pregunta**, sin ponderar por cantidad de relevantes. Las consultas sin ranking cuentan como cero. Se rechazan IDs desconocidos y duplicados en rankings para evitar resultados artificialmente inflados.

Ejemplo: relevantes `[b,c]`, ranking `[a,b,c]` y k=2 da Hit=1, Recall=0,5 y rango recíproco=0,5.

## Métodos incluidos

`overlap` cuenta palabras únicas compartidas. `tfidf` usa frecuencia bruta de término e IDF suavizado `log((N+1)/(df+1))+1`, con similitud coseno. No hay entrenamiento, stemming ni expansión de sinónimos. Ambos convierten a minúsculas, eliminan acentos y omiten una pequeña lista de palabras frecuentes en español.

Solo devuelven coincidencias con puntuación positiva; los empates se resuelven por ID de documento. Esto permite reproducir el ranking sin resultados aleatorios. Los scores entre métodos no se comparan directamente: se comparan sus métricas sobre las mismas preguntas.

## Salidas y límites

JSON y HTML muestran las métricas, los IDs recuperados y los relevantes que faltan por pregunta. El informe exporta **el texto de las consultas**, así que revísalo antes de compartirlo. No exporta el cuerpo de los documentos. Todo se calcula localmente y el HTML no carga recursos remotos.

Código de salida 0: informe generado. Código 2: entrada, parámetros o escritura inválidos. Un benchmark con métricas bajas sigue terminando en 0; no se ha definido un umbral automático. Las salidas existentes se reemplazan y no pueden coincidir con las entradas.

La relevancia depende de tus etiquetas: una lista incompleta puede penalizar un resultado válido. El conjunto pequeño de demostración no permite concluir que un método sea mejor en general. Estas métricas evalúan **recuperación**, no veracidad, citas ni calidad de respuestas generadas. No hay LLM juez, generación de respuestas ni conexión a una base vectorial.

## Pruebas

Las pruebas cubren métricas con valores conocidos, consultas sin resultados, validación de etiquetas e IDs, normalización de acentos, comparación externa, escape HTML, límites de k y ejecución real del CLI.
