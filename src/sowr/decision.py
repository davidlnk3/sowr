from functools import cache

from perplexity import Perplexity

DECIDER = "pplx-decider-flash-v1.2"


@cache
def _client():
    return Perplexity(timeout=30.0, max_retries=3)


def decide(state, questions, model=DECIDER):
    response = _client().decisions.create(
        model=model,
        state=state,
        questions=questions,
    )
    return response


def normalize(raw):
    """Reduce a Decider answer to {"answer", "confidence", "raw"}.

    Assumes a noul answer is the probability of "true".
    """
    match raw["type"]:
        case "choice":
            return {"answer": raw["choice"], "confidence": raw["confidence"], "raw": raw}
        case "score":
            return {"answer": raw["score"], "confidence": raw["confidence"], "raw": raw}
        case "noul":
            p = raw["noul"]
            return {"answer": "true" if p >= 0.5 else "false", "confidence": max(p, 1 - p), "raw": raw}


def answer_questions(questions, state, model=DECIDER):
    """Ask the Decider; return normalized answers by question id."""
    response = decide(state, questions, model)
    return {qid: normalize(a.model_dump()) for qid, a in response.answers.items()}


if __name__ == "__main__":
    state = {"review": "La batería dejó de cargar a las dos semanas."}
    questions = {
        "gravedad": {
            "type": "choice",
            "instructions": "¿Qué gravedad tiene el problema?",
            "criteria": {
                "Estético": "Solo afecta al aspecto del producto",
                "Molesto": "El producto funciona, pero con inconvenientes",
                "Producto inutilizable": "El producto ya no cumple su función",
            },
        },
    }
    response = decide(state, questions)
    print(response.to_json())
