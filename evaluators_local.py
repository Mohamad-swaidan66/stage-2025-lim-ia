# eval_rag/evaluators_local.py
import os, json, re, requests
from typing import Dict

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "llama3:latest")
TIMEOUT = int(os.getenv("RAG_JUDGE_TIMEOUT", "120"))

def _call_ollama_json(prompt: str) -> Dict:
    """
    Appelle Ollama en demandant une sortie JSON stricte.
    Force format=json et reconstruit l'objet JSON même s'il y a du texte autour.
    """
    body = {
        "model": JUDGE_MODEL,
        "prompt": prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no prose, no prefix.",
        "stream": False,
        "format": "json",  # si le modèle le supporte, il renverra uniquement du JSON
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }

    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    txt = (r.json().get("response") or "").strip()

    # 1) cas fréquent: réponses entourées de fences ```json ... ```
    if txt.startswith("```"):
        txt = txt.strip().strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:].lstrip("\r\n").lstrip("\n").strip("`").strip()

    # 2) tentative de parse direct
    try:
        return json.loads(txt)
    except Exception:
        pass

    # 3) fallback: extraire le premier objet JSON équilibré { ... }
    start = txt.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(txt)):
            ch = txt[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = txt[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break

    raise RuntimeError(f"Réponse juge non-JSON:\n{txt}")

# --------------- Prompts des 4 métriques ---------------

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Correctness: réponse vs ground truth (référence)."""
    prompt = f"""
Tu es un correcteur d'examen. Note uniquement l'exactitude factuelle de la réponse de l'élève
par rapport à la RÉPONSE DE RÉFÉRENCE (ground truth).
- Interdiction des contradictions.
- Des infos supplémentaires sont acceptables si factuellement cohérentes.

Rends UNIQUEMENT un JSON:
{{
  "explanation": "raisonnement concis",
  "correct": true/false
}}

QUESTION: {inputs['question']}
GROUND_TRUTH_ANSWER: {reference_outputs['answer']}
STUDENT_ANSWER: {outputs['answer']}
"""
    resp = _call_ollama_json(prompt)
    return bool(resp.get("correct", False))

def relevance(inputs: dict, outputs: dict) -> bool:
    """Relevance: la réponse adresse-t-elle la question ? (sans référence)"""
    prompt = f"""
Tu es un correcteur. Évalue si la réponse est concise et répond à la QUESTION.

Rends UNIQUEMENT un JSON:
{{
  "explanation": "raisonnement concis",
  "relevant": true/false
}}

QUESTION: {inputs['question']}
STUDENT_ANSWER: {outputs['answer']}
"""
    resp = _call_ollama_json(prompt)
    return bool(resp.get("relevant", False))

def groundedness(inputs: dict, outputs: dict) -> bool:
    """Groundedness: la réponse est-elle justifiée par les documents récupérés ?"""
    docs_text = "\n\n".join(getattr(d, "page_content", str(d)) for d in outputs["documents"])
    prompt = f"""
Tu es un correcteur. Vérifie que la réponse est entièrement fondée sur les FAITS ci-dessous,
sans ajouter d'informations absentes (pas d'hallucinations).

Rends UNIQUEMENT un JSON:
{{
  "explanation": "raisonnement concis",
  "grounded": true/false
}}

FAITS:
{docs_text}

RÉPONSE:
{outputs['answer']}
"""
    resp = _call_ollama_json(prompt)
    return bool(resp.get("grounded", False))

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """Retrieval relevance: les documents récupérés sont-ils pertinents à la question ?"""
    docs_text = "\n\n".join(getattr(d, "page_content", str(d)) for d in outputs["documents"])
    prompt = f"""
Tu es un correcteur. Évalue si les FAITS ci-dessous sont pertinents à la QUESTION.
- Si les FAITS contiennent des mots-clés ou du sens relié à la QUESTION, considère-les pertinents.
- Qu'ils contiennent aussi un peu d'information hors sujet est acceptable.

Rends UNIQUEMENT un JSON:
{{
  "explanation": "raisonnement concis",
  "relevant": true/false
}}

QUESTION:
{inputs['question']}

FAITS:
{docs_text}
"""
    resp = _call_ollama_json(prompt)
    return bool(resp.get("relevant", False))
