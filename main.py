import asyncio
import json
import logging
import os
import sys

import aiohttp
import openai
from nexus_client import NexusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

VK_URL = os.environ["VK_URL"]
MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
SERVICE_USERNAME = os.environ["MQTT_SERVICE_USERNAME"]
SERVICE_API_KEY = os.environ["MQTT_SERVICE_API_KEY"]
MNEMONIC_URL = os.environ["MNEMONIC_URL"]
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://thebrain.caronboulme.fr/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY = os.environ["LLAMACPP_API_KEY"]
HABIT_THRESHOLD = int(os.environ.get("HABIT_THRESHOLD", "5"))
# Un seul appel LLM sur tout le backlog de faits candidats (ex: 241 faits observés en
# pratique) génère un prompt énorme, lent (8-9 minutes mesurées) et sujet à des sorties
# dégénérées (le modèle boucle en répétant les mêmes IDs jusqu'à la limite de tokens, ou
# omet des champs requis). Regrouper par lots plus petits garde chaque appel rapide et
# fiable, tout en restant assez large pour dépasser HABIT_THRESHOLD au sein d'un même lot.
CONSOLIDATION_BATCH_SIZE = int(os.environ.get("CONSOLIDATION_BATCH_SIZE", "40"))

AGENT_NAME = "profiler"
HABIT_TTL_DAYS = int(os.environ.get("HABIT_TTL_DAYS", "30"))
HABIT_MATCH_THRESHOLD = float(os.environ.get("HABIT_MATCH_THRESHOLD", "0.35"))
# These fact types never expire — they describe who the person IS, not what they do
PERMANENT_TYPES = {"personal"}

_subscribed_users: set[str] = set()   # per-user: discussions subscription
_subscribed_sessions: set[str] = set()  # per-session: search/delete subscriptions
_user_passwords: dict[str, str] = {}  # username → current session cookie (refreshed on each reconnect)
_user_nexus: dict[str, object] = {}   # username → shared nexus for per-user subscriptions
_profile_tasks: dict[str, asyncio.Task] = {}  # username → pending debounced profile task
_consolidation_tasks: dict[str, asyncio.Task] = {}  # username → pending debounced consolidation task
_consolidation_running: dict[str, bool] = {}  # username → True once past the debounce delay, actively executing

# Taxonomie fermée : le champ 'type' d'un fait est contraint à ces 5 valeurs (voir
# _type_field_schema) et le LLM n'a JAMAIS la possibilité d'en inventer une nouvelle. Avant
# ce verrou, le champ était du texte libre avec une simple préférence pour réutiliser un type
# existant — sous température non nulle, le modèle dérivait vers des quasi-synonymes
# (interest / general_interest / literature_interest / book_interest...), fragmentant les
# mêmes faits sur des dizaines de types différents. Résultat concret : la sélection de type à
# la recherche devenait impossible à deviner de façon fiable, et la consolidation ne
# regroupait jamais assez de faits similaires pour dépasser HABIT_THRESHOLD.
DEFAULT_FACT_TYPES = ["assistant", "personal", "media", "interest", "places"]


def _type_field_schema(known_types: list[str]) -> dict:
    # known_types est ignoré à dessein : la taxonomie est fermée (voir DEFAULT_FACT_TYPES),
    # le paramètre reste pour ne pas toucher les appelants.
    return {
        "type": "string",
        "enum": DEFAULT_FACT_TYPES,
        "description": (
            "Fact category — always exactly one of these five, never invent a new one: "
            "'assistant' (how the user wants to interact with the assistant itself — tone, "
            "verbosity, jokes, language of the answers); "
            "'personal' (identity facts — name, family, friends, AND the user's OWN home "
            "address/city/place of residence); "
            "'media' (music listened to, contes/stories read or heard); "
            "'interest' (any other personal interest — news, food, video games, books other "
            "than contes, hobbies, sports...); "
            "'places' (places OTHER than the user's own home that are often searched or "
            "mentioned — restaurants, cities asked about for weather, addresses looked up, "
            "travel destinations)."
        ),
    }


def _make_extract_tool(known_types: list[str]) -> list:
    return [{
        "type": "function",
        "function": {
            "name": "extract_user_facts",
            "description": "Extract all personal facts and interests from the [user] lines in the conversation transcript.",
            "parameters": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": _type_field_schema(known_types),
                                "value": {"type": "string"},
                            },
                            "required": ["type", "value"],
                        },
                    }
                },
                "required": ["facts"],
            },
        },
    }]


def _make_consolidate_tool(known_types: list[str]) -> list:
    return [{
        "type": "function",
        "function": {
            "name": "declare_habits",
            "description": "Declare groups of similar facts that form a habit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "habits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "type": _type_field_schema(known_types),
                                "fact_ids": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["description", "type", "fact_ids"],
                        },
                    }
                },
                "required": ["habits"],
            },
        },
    }]


def _find_topic(private_topics: list, suffix: str) -> str | None:
    for agent_entry in private_topics:
        for t in agent_entry.get("topics", []):
            if t["topic"].endswith(f"/{suffix}"):
                return t["topic"]
    return None


EXTRACT_SYSTEM_PROMPT = (
    "You extract personal facts and interests about the human user from a conversation transcript. "
    "The transcript contains [user] and [assistant] lines. "
    "Extract facts ONLY from [user] lines. Use [assistant] lines as context to better understand and categorize user messages. "
    "Every [user] line reveals at least one fact: questions reveal interests, requests reveal needs, statements reveal preferences. "
    "Values must be complete English statements, never French. "
    "Categorize EVERY fact into EXACTLY one of five fixed categories — NEVER invent a new one:\n"
    "- assistant: how the user wants to interact with the assistant itself (tone, verbosity, jokes, language of the answers).\n"
    "- personal: identity facts — name, family, friends, AND the user's OWN home address/city/place of residence.\n"
    "- media: music listened to, contes/stories read or heard.\n"
    "- interest: any other personal interest — news, food, video games, books other than contes, hobbies, sports...\n"
    "- places: places OTHER than the user's own home that are often searched or mentioned — restaurants, cities asked about for weather, addresses looked up, travel destinations.\n"
    "IGNORE completely: any user message asking to forget, delete, erase or clear memories/facts/data. "
    "These are instructions to the system, not personal facts to record. Do NOT extract them. "
    "Examples:\n"
    "- [user]: je m'appelle François → {type: \"personal\", value: \"is named François\"}\n"
    "- [user]: mon prénom c'est Marie → {type: \"personal\", value: \"is named Marie\"}\n"
    "- [user]: j'habite à Lyon → {type: \"personal\", value: \"lives in Lyon\"} (this is the user's OWN residence, always 'personal', never 'places')\n"
    "- [user]: quelle meteo demain a paris → {type: \"places\", value: \"asks about weather in Paris\"} (Paris here is NOT the user's home — a place they're merely asking about)\n"
    "- [user]: j aime le retro gaming → {type: \"interest\", value: \"likes retro gaming\"}\n"
    "- [user]: tu connais le cycle de Hain? / [assistant]: c'est une série SF / [user]: c'est plusieurs livres → {type: \"interest\", value: \"is interested in the Hain cycle\"}\n"
    "- [user]: raconte-moi Pinocchio → {type: \"media\", value: \"listens to the story of Pinocchio\"}\n"
    "- [user]: mets-moi de la musique de Bach → {type: \"media\", value: \"listens to Bach\"}\n"
    "- [user]: réponds-moi plus brièvement à l'avenir → {type: \"assistant\", value: \"prefers concise responses\"}\n"
    "- [user]: oublie tout sur moi → (nothing to extract, this is a deletion command)\n"
    "- [user]: efface mes données → (nothing to extract, this is a deletion command)\n"
    "Call extract_user_facts with ALL facts found."
)


def _extract_facts_sync(messages: list, known_types: list[str]) -> list[dict]:
    transcript = "\n".join(
        f"[{m['role']}]: {m['content']}"
        for m in messages if m.get("role") in ("user", "assistant")
    )
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation:\n{transcript}"},
            ],
            tools=_make_extract_tool(known_types),
            tool_choice="required",
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return []
        return json.loads(tool_calls[0].function.arguments).get("facts", [])
    except Exception as e:
        logger.error(f"Extraction de faits échouée: {e}")
        return []


async def _fetch_known_types(username: str, auth_headers: dict) -> list[str]:
    try:
        async with aiohttp.ClientSession(headers=auth_headers) as http:
            resp = await http.get(f"{MNEMONIC_URL}/users/{username}/facts/types")
            resp.raise_for_status()
            types = (await resp.json()).get("types", [])
            return types if types else DEFAULT_FACT_TYPES
    except Exception as e:
        logger.warning(f"[{username}] Impossible de récupérer les types, utilisation des défauts: {e}")
        return DEFAULT_FACT_TYPES


async def _extract_facts(messages: list, known_types: list[str]) -> list[dict]:
    user_count = sum(1 for m in messages if m.get("role") == "user")
    logger.info(f"LLM POST {LLM_BASE_URL}/chat/completions — model={LLM_MODEL}, {user_count} messages utilisateur sur {len(messages)}, types connus: {known_types}")
    logger.info(f"System prompt: {EXTRACT_SYSTEM_PROMPT}")
    loop = asyncio.get_event_loop()
    facts = await loop.run_in_executor(None, _extract_facts_sync, messages, known_types)
    logger.info(f"Faits extraits: {json.dumps(facts, ensure_ascii=False)}")
    return facts


def _find_habits_sync(facts: list[dict], known_types: list[str]) -> list[dict]:
    if len(facts) < HABIT_THRESHOLD:
        return []
    facts_text = "\n".join(f"- id={f['id']} type={f['type']} value=\"{f['value']}\"" for f in facts)
    logger.info(f"Consolidation — LLM POST {LLM_BASE_URL}/chat/completions avec {len(facts)} faits")
    logger.info(f"Consolidation — faits envoyés:\n{facts_text}")
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You analyze a list of user facts and identify groups of similar facts that reveal a recurring habit. "
                    f"Only group facts that are clearly related AND have at least {HABIT_THRESHOLD} facts in the group (e.g. {HABIT_THRESHOLD}+ weather requests = habit of checking weather). "
                    "For each group, write a short English habit description (e.g. 'regularly checks weather forecasts', 'passionate about retro gaming'). "
                    "Call declare_habits with the groups found. If no groups exist, call declare_habits with an empty list."
                )},
                {"role": "user", "content": f"Facts:\n{facts_text}"},
            ],
            tools=_make_consolidate_tool(known_types),
            tool_choice="required",
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            logger.warning("Consolidation — LLM n'a pas retourné de tool call")
            return []
        result = json.loads(tool_calls[0].function.arguments).get("habits", [])
        logger.info(f"Consolidation — LLM response: {json.dumps(result, ensure_ascii=False)}")
        return result
    except Exception as e:
        logger.error(f"Identification des habitudes échouée: {e}")
        return []


def _habit_expires_at() -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(days=HABIT_TTL_DAYS)).isoformat()


async def _match_facts_to_habits(username: str, facts: list[dict], auth_headers: dict) -> tuple[list[dict], list[dict]]:
    """Split facts into (unmatched → store as new facts, matched → refresh existing habit).
    Returns (facts_to_store, []) — habit refreshes are done in-place."""
    if not facts:
        return facts, []
    try:
        async with aiohttp.ClientSession(headers=auth_headers) as http:
            facts_to_store = []
            for fact in facts:
                resp = await http.get(
                    f"{MNEMONIC_URL}/users/{username}/habits/search",
                    params={"q": fact["value"], "n": 1, "threshold": HABIT_MATCH_THRESHOLD},
                )
                resp.raise_for_status()
                matches = await resp.json()
                if matches:
                    habit = matches[0]
                    fact_type = fact.get("type", "")
                    if fact_type in PERMANENT_TYPES:
                        # Permanent types: never expire, no refresh needed
                        logger.info(f"[{username}] Fait permanent ignoré (correspond à habitude {habit['id']}): {fact['value']!r}")
                    else:
                        new_expires = _habit_expires_at()
                        await http.patch(
                            f"{MNEMONIC_URL}/users/{username}/habits/{habit['id']}",
                            params={"expires_at": new_expires},
                        )
                        logger.info(f"[{username}] Habitude {habit['id']!r} rafraîchie ({fact['value']!r} ≈ {habit['value']!r}, distance={habit['distance']:.3f})")
                else:
                    facts_to_store.append(fact)
            return facts_to_store, []
    except Exception as e:
        logger.error(f"[{username}] Matching habitudes échoué: {e}")
        return facts, []


async def _consolidate_habits(username: str, auth_headers: dict):
    logger.info(f"[{username}] Consolidation des habitudes en cours...")
    try:
        async with aiohttp.ClientSession(headers=auth_headers) as http:
            resp = await http.get(f"{MNEMONIC_URL}/users/{username}/facts")
            resp.raise_for_status()
            all_facts = await resp.json()
    except Exception as e:
        logger.error(f"[{username}] Échec récupération faits pour consolidation: {e}")
        return

    # Only consolidate non-permanent types — permanent ones are kept as individual facts
    non_permanent_facts = [f for f in all_facts if f["type"] not in PERMANENT_TYPES]
    logger.info(f"[{username}] {len(non_permanent_facts)}/{len(all_facts)} faits non-permanents (seuil: {HABIT_THRESHOLD})")

    type_counts = {}
    for f in non_permanent_facts:
        type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1

    candidate_facts = [f for f in non_permanent_facts if type_counts[f["type"]] >= HABIT_THRESHOLD]
    if not candidate_facts:
        logger.info(f"[{username}] Aucun type avec ≥{HABIT_THRESHOLD} faits, pas de consolidation")
        return
    logger.info(f"[{username}] {len(candidate_facts)} faits candidats (types: {[t for t, c in type_counts.items() if c >= HABIT_THRESHOLD]})")

    known_types = sorted(set(f["type"] for f in all_facts)) or DEFAULT_FACT_TYPES
    loop = asyncio.get_event_loop()

    # Lots homogènes par type (des faits de types différents n'ont de toute façon aucune
    # raison d'être regroupés ensemble), puis découpés à CONSOLIDATION_BATCH_SIZE — voir le
    # commentaire sur la constante pour le pourquoi.
    facts_by_type: dict[str, list[dict]] = {}
    for f in candidate_facts:
        facts_by_type.setdefault(f["type"], []).append(f)
    batches = [
        facts_of_type[i:i + CONSOLIDATION_BATCH_SIZE]
        for facts_of_type in facts_by_type.values()
        for i in range(0, len(facts_of_type), CONSOLIDATION_BATCH_SIZE)
    ]
    logger.info(f"[{username}] Regroupement en {len(batches)} lot(s) de ≤{CONSOLIDATION_BATCH_SIZE} faits")

    batch_results = await asyncio.gather(
        *[loop.run_in_executor(None, _find_habits_sync, batch, known_types) for batch in batches]
    )
    habit_groups = [group for groups in batch_results for group in groups]

    if not habit_groups:
        logger.info(f"[{username}] Aucune habitude détectée")
        return

    logger.info(f"[{username}] {len(habit_groups)} habitude(s) détectée(s)")
    facts_by_id = {f["id"]: f for f in all_facts}

    for habit in habit_groups:
        # Un groupe malformé (clé manquante/mauvais type — déjà vu en pratique quand le LLM
        # part en boucle de répétition sur fact_ids jusqu'à la limite de tokens) ne doit PAS
        # faire échouer tout le cycle en silence : sans ce try/except, une exception ici
        # remontait hors d'une asyncio.create_task() que personne n'attend, donc jamais
        # loggée (asyncio ne la signale qu'au garbage collection de la tâche, qui n'arrive
        # jamais tant qu'elle reste référencée dans _consolidation_tasks) — la consolidation
        # semblait alors juste s'arrêter net, sans aucune trace ni erreur.
        try:
            fact_ids = habit["fact_ids"]
            description = habit["description"]
            habit_type = habit["type"]
            valid_ids = [fid for fid in fact_ids if fid in facts_by_id]
        except (KeyError, TypeError) as e:
            logger.error(f"[{username}] Groupe d'habitude malformé, ignoré: {e} — habit={repr(habit)[:500]}")
            continue

        logger.info(f"[{username}] Habitude '{description}': {len(valid_ids)} faits valides sur {len(fact_ids)} proposés")
        if len(valid_ids) < HABIT_THRESHOLD:
            logger.info(f"[{username}] Ignoré — moins de {HABIT_THRESHOLD} faits valides")
            continue

        is_permanent = habit_type in PERMANENT_TYPES
        expires_at = "" if is_permanent else _habit_expires_at()

        session_ids = list(dict.fromkeys(facts_by_id[fid]["session_id"] for fid in valid_ids))
        logger.info(f"[{username}] Stockage habitude: type={habit_type} permanent={is_permanent} description=\"{description}\"")

        try:
            async with aiohttp.ClientSession(headers=auth_headers) as http:
                resp = await http.post(
                    f"{MNEMONIC_URL}/users/{username}/facts",
                    json={
                        "facts": [{"type": habit_type, "value": habit["description"]}],
                        "session_id": session_ids[0],
                        "session_ids": session_ids,
                        "is_habit": True,
                        "expires_at": expires_at,
                    },
                )
                resp.raise_for_status()
                logger.info(f"[{username}] Habitude stockée (expires: {expires_at or 'jamais'})")

                for fid in valid_ids:
                    del_resp = await http.delete(f"{MNEMONIC_URL}/users/{username}/facts/{fid}")
                    logger.info(f"[{username}] Fait {fid} supprimé (status {del_resp.status})")

            logger.info(f"[{username}] Consolidation terminée: {len(valid_ids)} faits → 1 habitude")
        except Exception as e:
            logger.error(f"[{username}] Échec consolidation habitude: {e}")


def _schedule_consolidation(username: str):
    """Debounce _consolidate_habits comme _schedule_profile_generation le fait déjà pour
    le profil : sans ça, chaque discussion déclenchait sa propre passe de consolidation
    complète (re-scan de TOUS les faits + un appel LLM sur l'ensemble), et une rafale de
    discussions (ex: plusieurs sessions de test rapprochées) empilait des passes redondantes
    et de plus en plus coûteuses au lieu d'une seule consolidation sur l'état final."""
    existing = _consolidation_tasks.get(username)
    if existing and not existing.done():
        if _consolidation_running.get(username):
            # Une passe est déjà en cours d'exécution (au-delà du délai de debounce — LLM de
            # regroupement puis stockage/suppression) : ne PAS l'annuler. Avant ce garde-fou,
            # une discussion arrivant pendant cette fenêtre annulait la tâche en plein vol,
            # parfois après que le LLM ait déjà identifié un groupe de faits en double (log
            # "N habitude(s) détectée(s)"), coupant court avant le stockage de l'habitude et
            # la suppression des faits d'origine. Résultat observé en pratique : les faits
            # quasi-identiques s'accumulaient indéfiniment (des dizaines de doublons) sans
            # jamais être consolidés, et search_preference devenait de plus en plus lent
            # (il passe tous les faits d'un type au LLM de synthèse). On laisse cette passe
            # se terminer ; la prochaine discussion en planifiera une nouvelle une fois
            # celle-ci achevée.
            logger.debug(f"[{username}] Consolidation déjà en cours d'exécution — nouvelle planification ignorée")
            return
        existing.cancel()
        logger.debug(f"[{username}] Consolidation annulée (debounce)")

    async def _delayed():
        await asyncio.sleep(2.0)
        _consolidation_running[username] = True
        try:
            auth_headers = {"Cookie": f"vk_session={_user_passwords[username]}"}
            await _consolidate_habits(username, auth_headers)
        except Exception as e:
            # Filet de sécurité : la tâche créée par asyncio.create_task() n'est jamais
            # attendue nulle part, donc une exception non catchée ici resterait invisible
            # (asyncio ne la loggue qu'au garbage collection de la tâche, qui n'arrive pas
            # tant qu'elle reste référencée dans _consolidation_tasks) — déjà vécu une fois
            # avec une consolidation qui semblait juste s'arrêter net sans aucune trace.
            logger.error(f"[{username}] Consolidation échouée (exception non gérée): {e}", exc_info=True)
        finally:
            _consolidation_running[username] = False

    _consolidation_tasks[username] = asyncio.create_task(_delayed())
    logger.info(f"[{username}] Consolidation planifiée (debounce 2s)")


def _select_search_types_sync(query: str, available_types: list[str]) -> list[str]:
    if not available_types:
        return []
    tool = [{
        "type": "function",
        "function": {
            "name": "select_types",
            "description": "Select the fact types relevant to the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types from the available list that would contain facts answering the query.",
                    }
                },
                "required": ["types"],
            },
        },
    }]
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Given a search query about a user's preferences or personal data, "
                    "select the fact types from the available list that are most likely to contain relevant facts. "
                    "Example: query='villes préférées' → types=['location']. "
                    "Call select_types with the matching types."
                )},
                {"role": "user", "content": f"Query: {query}\nAvailable types: {', '.join(available_types)}"},
            ],
            tools=tool,
            tool_choice="required",
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return []
        result = json.loads(tool_calls[0].function.arguments).get("types", [])
        logger.info(f"Types sélectionnés pour la recherche: {result}")
        return [t for t in result if t in available_types]
    except Exception as e:
        logger.error(f"Sélection des types de recherche échouée: {e}")
        return []


def _select_deletion_types_sync(query: str, available_types: list[str]) -> list[str]:
    """Like _select_search_types_sync but strict: only types whose facts explicitly name the subject."""
    if not available_types:
        return []
    tool = [{
        "type": "function",
        "function": {
            "name": "select_types",
            "description": "Select fact types whose stored facts would explicitly mention the deletion subject.",
            "parameters": {
                "type": "object",
                "properties": {
                    "types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Only the types whose facts would directly name the subject. "
                            "Do NOT select associative types (e.g. for 'Paris weather' → ['location'] only, "
                            "NOT cuisine/cinema/philosophy even if Paris is associated with them)."
                        ),
                    }
                },
                "required": ["types"],
            },
        },
    }]
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "For a deletion query, select ONLY the 1 or 2 fact types that would contain facts "
                    "directly and explicitly naming the subject. "
                    "Think: which type stores a fact whose value would literally contain the subject word? "
                    "Ignore cultural or thematic associations. "
                    "Examples:\n"
                    "- query='Paris weather' → ['location'] (a location fact might say 'is interested in weather in Paris')\n"
                    "- query='retro gaming' → ['video_game']\n"
                    "- query='François' → ['name', 'person']\n"
                    "Call select_types with all directly relevant types."
                )},
                {"role": "user", "content": f"Query: {query}\nAvailable types: {', '.join(available_types)}"},
            ],
            tools=tool,
            tool_choice="required",
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return []
        result = json.loads(tool_calls[0].function.arguments).get("types", [])
        filtered = [t for t in result if t in available_types]
        logger.info(f"Types sélectionnés pour suppression: {filtered}")
        return filtered
    except Exception as e:
        logger.error(f"Sélection des types de suppression échouée: {e}")
        return []


def _build_profile_sync(username: str, personal_facts: list[dict], habits: list[dict]) -> str:
    lines = []
    if personal_facts:
        lines.append("Known facts:")
        for f in personal_facts:
            lines.append(f"- {f['value']}")
    if habits:
        lines.append("Recurring interests/habits:")
        for h in habits:
            lines.append(f"- {h['value']}")
    profile = "\n".join(lines)
    logger.info(f"[{username}] Profil construit:\n{profile}")
    return profile


def _filter_facts_for_deletion_sync(query: str, facts: list[dict]) -> list[str]:
    if not facts:
        return []
    facts_text = "\n".join(f"- id={f['id']} type={f['type']} value=\"{f['value']}\"" for f in facts)
    tool = [{
        "type": "function",
        "function": {
            "name": "select_facts_to_delete",
            "description": "Select the IDs of facts that match the deletion query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "IDs of facts that directly and explicitly match the query. "
                            "Be conservative: only include facts that clearly reference the subject. "
                            "If unsure, exclude."
                        ),
                    }
                },
                "required": ["ids"],
            },
        },
    }]
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are given a deletion query and a list of user facts. "
                    "Select ONLY the IDs of facts that match BOTH the subject AND the intent of the query. "
                    "Be conservative — when in doubt, do NOT include the fact. "
                    "Examples:\n"
                    "- query='météo sur Paris' → select facts about weather interest in Paris "
                    "(e.g. 'is interested in weather in Paris'), NOT facts about living in Paris or Paris being a favourite city.\n"
                    "- query='Marseille' (no context) → select facts that mention Marseille in any context.\n"
                    "- query='retro gaming' → select facts about retro gaming interest only, not general gaming facts."
                )},
                {"role": "user", "content": f"Query: {query}\n\nFacts:\n{facts_text}"},
            ],
            tools=tool,
            tool_choice="required",
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return []
        ids = json.loads(tool_calls[0].function.arguments).get("ids", [])
        valid_ids = {f["id"] for f in facts}
        return [fid for fid in ids if fid in valid_ids]
    except Exception as e:
        logger.error(f"Filtrage suppression échoué: {e}")
        return []


def _synthesize_search_sync(query: str, facts: list[dict]) -> str:
    if not facts:
        return "Aucun résultat trouvé."
    facts_text = "\n".join(f"- [{f['type']}] {f['value']}" for f in facts)
    tool = [{
        "type": "function",
        "function": {
            "name": "report_answer",
            "description": "Report the answer extracted from the facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "The values from the facts that directly answer the query, as a short comma-separated list. "
                            "If no facts are relevant, return 'Aucune information trouvée.'"
                        ),
                    }
                },
                "required": ["answer"],
            },
        },
    }]
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are given stored facts about a user and a query. "
                    "Call report_answer with the values that directly answer the query. "
                    "Example: query='favourite sports' facts=[sport: football, sport: tennis] → answer='football, tennis'"
                )},
                {"role": "user", "content": f"Query: {query}\n\nFacts:\n{facts_text}"},
            ],
            tools=tool,
            tool_choice="required",
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            logger.warning("Synthèse: pas de tool call, fallback sur les valeurs brutes")
            return ", ".join(f["value"] for f in facts)
        answer = json.loads(tool_calls[0].function.arguments).get("answer", "")
        logger.info(f"Synthèse LLM — answer={answer!r}")
        return answer or ", ".join(f["value"] for f in facts)
    except Exception as e:
        logger.error(f"Synthèse résultats échouée: {e}")
        return ", ".join(f["value"] for f in facts)


async def _generate_profile(username: str, auth_headers: dict, nexus, profile_topic: str):
    logger.info(f"[{username}] Génération du profil...")
    loop = asyncio.get_event_loop()

    # Avec la taxonomie fermée, les faits qui décrivent qui EST la personne (par
    # opposition à ses habitudes) sont toujours exactement PERMANENT_TYPES ('personal') —
    # plus besoin d'un appel LLM pour le deviner parmi une liste de types dynamique comme
    # à l'époque de l'ancienne taxonomie ouverte.
    profile_types = sorted(PERMANENT_TYPES)

    personal_facts = []
    habits = []
    try:
        async with aiohttp.ClientSession(headers=auth_headers) as http:
            for fact_type in profile_types:
                resp = await http.get(f"{MNEMONIC_URL}/users/{username}/facts", params={"fact_type": fact_type})
                resp.raise_for_status()
                personal_facts.extend(await resp.json())
            resp = await http.get(f"{MNEMONIC_URL}/users/{username}/habits")
            resp.raise_for_status()
            habits = await resp.json()
    except Exception as e:
        logger.error(f"[{username}] Échec récupération faits/habitudes pour profil: {e}")
        return

    logger.info(f"[{username}] {len(personal_facts)} faits personnels, {len(habits)} habitudes")
    if not personal_facts and not habits:
        logger.info(f"[{username}] Rien à profiler")
        return

    profile_text = await loop.run_in_executor(None, _build_profile_sync, username, personal_facts, habits)
    if not profile_text:
        return

    logger.info(f"[{username}] Profil généré:\n{profile_text}")
    await nexus.publish(
        profile_topic,
        {"username": username, "summary": profile_text},
        retain=True,
    )
    logger.info(f"[{username}] Profil publié sur {profile_topic}")


def _schedule_profile_generation(username: str, nexus, profile_topic: str):
    existing = _profile_tasks.get(username)
    if existing and not existing.done():
        existing.cancel()
        logger.debug(f"[{username}] Régénération profil annulée (debounce)")

    async def _delayed():
        await asyncio.sleep(0.5)
        auth_headers = {"Cookie": f"vk_session={_user_passwords[username]}"}
        await _generate_profile(username, auth_headers, nexus, profile_topic)

    _profile_tasks[username] = asyncio.create_task(_delayed())
    logger.info(f"[{username}] Régénération profil planifiée (debounce 500ms)")


def _normalize_messages(payload: list) -> list:
    """Convert multimodal messages to text-only for mnemonic storage."""
    normalized = []
    for msg in payload:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(parts).strip()
        if not isinstance(content, str):
            content = str(content)
        normalized.append({"role": msg.get("role", "user"), "content": content})
    return normalized


async def on_discussion(username: str, topic: str, payload, user_api_key: str, nexus, profile_topic: str):
    if not isinstance(payload, list) or not payload:
        return

    logger.info(f"[{username}] Discussion reçue ({len(payload)} messages)")

    auth_headers = {"Cookie": f"vk_session={user_api_key}"}
    sessions_url = f"{MNEMONIC_URL}/users/{username}/sessions"
    logger.info(f"[{username}] POST {sessions_url} — Cookie: vk_session={user_api_key}")

    normalized = _normalize_messages(payload)
    if not normalized:
        logger.info(f"[{username}] Aucun message texte après normalisation, skip")
        return

    try:
        async with aiohttp.ClientSession(headers=auth_headers) as http:
            resp = await http.post(
                sessions_url,
                json={"messages": normalized},
            )
            resp.raise_for_status()
            session_id = (await resp.json())["session_id"]
        logger.info(f"[{username}] Session {session_id} stockée dans mnemonic")
    except Exception as e:
        logger.error(f"[{username}] Échec stockage session dans mnemonic: {e}")
        return

    known_types = await _fetch_known_types(username, auth_headers)
    logger.info(f"[{username}] Types connus: {known_types}")

    logger.info(f"[{username}] Extraction des faits en cours...")
    facts = await _extract_facts(normalized, known_types)
    if not facts:
        logger.info(f"[{username}] Aucun fait extrait")
        return

    logger.info(f"[{username}] {len(facts)} faits extraits:")
    for fact in facts:
        logger.info(f"[{username}]   {fact['type']}: {fact['value']}")

    # Filter out facts that already match an existing habit (refresh the habit instead)
    facts_to_store, _ = await _match_facts_to_habits(username, facts, auth_headers)
    logger.info(f"[{username}] {len(facts_to_store)}/{len(facts)} faits à stocker (autres = habitudes rafraîchies)")

    if facts_to_store:
        try:
            async with aiohttp.ClientSession(headers=auth_headers) as http:
                resp = await http.post(
                    f"{MNEMONIC_URL}/users/{username}/facts",
                    json={"facts": facts_to_store, "session_id": session_id},
                )
                resp.raise_for_status()
            logger.info(f"[{username}] Faits enregistrés dans mnemonic")
        except Exception as e:
            logger.error(f"[{username}] Échec enregistrement des faits dans mnemonic: {e}")
            return

    _schedule_consolidation(username)
    _schedule_profile_generation(username, nexus, profile_topic)


async def on_user_connected(topic: str, payload):
    if not isinstance(payload, dict):
        return

    username = payload.get("username")
    password = payload.get("password")
    session_id = payload.get("session_id")
    private_topics = payload.get("private_topics", [])

    if not username or not password or not session_id:
        return

    discussions_topic = _find_topic(private_topics, "discussions")
    agent_topics_topic = _find_topic(private_topics, "agent_topics")

    if not discussions_topic or not agent_topics_topic:
        logger.warning(f"Topics manquants pour {username}, skip")
        return

    # Per-user: profile and discussions stay per-user
    profile_topic = f"users/{username}/profile"

    # Per-session: search/delete topics scoped to this connection
    search_topic = f"users/{username}/{session_id}/search_preference"
    delete_topic = f"users/{username}/{session_id}/delete_facts"
    search_results_topic = f"users/{username}/{session_id}/search_results"
    delete_results_topic = f"users/{username}/{session_id}/delete_results"

    # Reuse per-user nexus for discussions (avoid duplicate subscriptions across sessions)
    if username not in _user_nexus:
        _user_nexus[username] = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)
    nexus = _user_nexus[username]

    # Always republish topic declaration so agents that restarted can rediscover it.
    # retain=True ensures late-subscribing backends (e.g. after restart) still receive it.
    await nexus.publish(
        agent_topics_topic,
        [{
            "agent": AGENT_NAME,
            "topics": [
                {
                    "topic": profile_topic,
                    "description": "Profil utilisateur",
                    "access": "read",
                    "format": {"username": "string", "summary": "string"},
                },
                {
                    "topic": search_topic,
                    "description": (
                        "Recherche dans la mémoire de l'utilisateur (données/préférences/habitudes "
                        "propres, ex: 'mes lieux habituels', 'est-ce que j'aime le jazz'), y compris "
                        "comme étape préalable avant une autre action (ex: 'météo de mes lieux "
                        "habituels' → chercher d'abord 'lieux habituels')."
                    ),
                    "access": "write",
                    "response_topic": search_results_topic,
                    "format": {"query": "string", "n": 5},
                },
                {
                    "topic": delete_topic,
                    "description": (
                        "Supprime des faits mémorisés — demande explicite (supprimer/effacer/oublier) "
                        "OU déclaration de désintérêt (ex: 'la météo ne m'intéresse plus'). PAS pour un "
                        "commentaire passager dans une demande d'action ('donne-moi la météo' ne "
                        "supprime rien). clear_all=true pour tout effacer ('oublie tout')."
                    ),
                    "access": "write",
                    "response_topic": delete_results_topic,
                    "format": {"query": "string (OR) ids: [\"...\"] (OR) clear_all: true"},
                },
                {
                    "topic": search_results_topic,
                    "description": "Réponse synthétisée à la dernière recherche de faits",
                    "access": "read",
                    "format": {"query": "string", "answer": "string"},
                },
                {
                    "topic": delete_results_topic,
                    "description": "Résultat de la dernière suppression de faits",
                    "access": "read",
                    "format": {"query": "string", "deleted_count": "int", "deleted": ["string"]},
                },
            ],
        }],
        retain=True,
    )
    logger.info(f"[{username}/{session_id}] Topics déclarés sur {agent_topics_topic}")

    # Always refresh the password — cookie expires after 24h
    _user_passwords[username] = password
    logger.info(f"[{username}] Cookie de session mis à jour")

    # Per-user: subscribe to discussions only once
    if username not in _subscribed_users:
        _subscribed_users.add(username)
        logger.info(f"Nouvel utilisateur: {username} — discussions={discussions_topic}")

        async def handler(t, p):
            await on_discussion(username, t, p, _user_passwords[username], nexus, profile_topic)

        nexus.subscribe(discussions_topic, handler)

    # Per-session: subscribe to search/delete
    if session_id in _subscribed_sessions:
        nexus.start_listening()
        logger.debug(f"[{username}/{session_id}] Session déjà abonnée, skip search/delete")
        return

    _subscribed_sessions.add(session_id)
    logger.info(f"[{username}/{session_id}] Nouvelle session — abonnement search/delete")

    async def on_search_request(t, p):
        if not isinstance(p, dict):
            return
        # Normalize keys: Gemma4 sometimes generates escaped quotes around key names
        p = {k.strip('"'): v for k, v in p.items()}
        query = p.get("query", "")
        n = int(p.get("n", 5))
        if not query:
            return
        logger.info(f"[{username}] Recherche de faits: {query!r} (n={n})")

        loop = asyncio.get_event_loop()

        # 1. Fetch available types
        available_types = await _fetch_known_types(username, {"Cookie": f"vk_session={_user_passwords[username]}"})
        logger.info(f"[{username}] Types disponibles: {available_types}")

        # 2. LLM selects relevant types for this query — sans type sélectionné (question
        # trop large pour cibler, ex: "que sais-tu sur moi ?"), chercher sur tous les types
        # disponibles plutôt que de se rabattre uniquement sur la recherche sémantique : avec
        # seulement 5 catégories fixes désormais, interroger les 5 ne coûte rien et garantit
        # de ne rater aucune donnée par excès de prudence du sélecteur.
        selected_types = await loop.run_in_executor(
            None, _select_search_types_sync, query, available_types
        )
        if not selected_types:
            selected_types = available_types
        logger.info(f"[{username}] Types retenus pour la recherche: {selected_types}")

        # 3. Fetch facts by type; fall back to semantic search if no types matched
        facts = []
        if selected_types:
            try:
                async with aiohttp.ClientSession(headers={"Cookie": f"vk_session={_user_passwords[username]}"}) as http:
                    for fact_type in selected_types:
                        resp = await http.get(
                            f"{MNEMONIC_URL}/users/{username}/facts",
                            params={"fact_type": fact_type},
                        )
                        resp.raise_for_status()
                        facts.extend(await resp.json())
                logger.info(f"[{username}] {len(facts)} faits récupérés par type: {facts}")
            except Exception as e:
                logger.error(f"[{username}] Échec récupération par type: {e}")

        if not facts:
            logger.info(f"[{username}] Fallback sur la recherche sémantique")
            try:
                async with aiohttp.ClientSession(headers={"Cookie": f"vk_session={_user_passwords[username]}"}) as http:
                    resp = await http.get(
                        f"{MNEMONIC_URL}/users/{username}/facts/search",
                        params={"q": query, "n": n},
                    )
                    resp.raise_for_status()
                    facts = await resp.json()
                logger.info(f"[{username}] Mnemonic résultats sémantiques ({len(facts)}): {facts}")
            except Exception as e:
                logger.error(f"[{username}] Échec recherche sémantique: {e}")
                return

        # 4. LLM synthesizes a focused answer
        answer = await loop.run_in_executor(None, _synthesize_search_sync, query, facts)
        logger.info(f"[{username}] Synthèse: {answer!r}")
        await nexus.publish(search_results_topic, {"query": query, "answer": answer})
        logger.info(f"[{username}] Résultats publiés sur {search_results_topic}")

    async def on_delete_request(t, p):
        if not isinstance(p, dict):
            return
        p = {k.strip('"'): v for k, v in p.items()}
        ids = p.get("ids")
        query = p.get("query", "")
        clear_all = p.get("clear_all", False)
        if not ids and not query and not clear_all:
            return

        deleted_labels = []

        if clear_all:
            logger.info(f"[{username}] Suppression totale de la mémoire")
            try:
                async with aiohttp.ClientSession(headers={"Cookie": f"vk_session={_user_passwords[username]}"}) as http:
                    resp = await http.delete(f"{MNEMONIC_URL}/users/{username}/facts")
                    resp.raise_for_status()
                    result = await resp.json()
                    n = result.get("deleted_count", 0)
                    deleted_labels = [f"all ({n} facts)"]
                    logger.info(f"[{username}] Mémoire effacée: {n} faits supprimés")
            except Exception as e:
                logger.error(f"[{username}] Échec suppression totale: {e}")
        elif ids:
            logger.info(f"[{username}] Suppression par ids: {ids}")
            async with aiohttp.ClientSession(headers={"Cookie": f"vk_session={_user_passwords[username]}"}) as http:
                for fact_id in ids:
                    try:
                        resp = await http.delete(f"{MNEMONIC_URL}/users/{username}/facts/{fact_id}")
                        resp.raise_for_status()
                        deleted_labels.append(fact_id)
                        logger.info(f"[{username}] Fait supprimé: {fact_id}")
                    except Exception as e:
                        logger.error(f"[{username}] Échec suppression {fact_id}: {e}")
        else:
            logger.info(f"[{username}] Suppression par recherche: {query!r}")

            loop = asyncio.get_event_loop()

            # 1. Find directly relevant types for this query (strict, max 2)
            available_types = await _fetch_known_types(username, {"Cookie": f"vk_session={_user_passwords[username]}"})
            selected_types = await loop.run_in_executor(
                None, _select_deletion_types_sync, query, available_types
            )
            logger.info(f"[{username}] Types retenus pour suppression: {selected_types}")

            # 2. Fetch candidate facts by type
            candidates = []
            if selected_types:
                try:
                    async with aiohttp.ClientSession(headers={"Cookie": f"vk_session={_user_passwords[username]}"}) as http:
                        for fact_type in selected_types:
                            resp = await http.get(
                                f"{MNEMONIC_URL}/users/{username}/facts",
                                params={"fact_type": fact_type},
                            )
                            resp.raise_for_status()
                            candidates.extend(await resp.json())
                    logger.info(f"[{username}] {len(candidates)} candidats à la suppression")
                except Exception as e:
                    logger.error(f"[{username}] Échec récupération candidats: {e}")
                    await nexus.publish(delete_results_topic, {"query": query, "deleted_count": 0, "deleted": []})
                    return

            if not candidates:
                logger.info(f"[{username}] Aucun candidat trouvé pour suppression: {query!r}")
                await nexus.publish(delete_results_topic, {"query": query, "deleted_count": 0, "deleted": []})
                return

            # 3. LLM filters to only facts that truly match the query
            ids_to_delete = await loop.run_in_executor(
                None, _filter_facts_for_deletion_sync, query, candidates
            )
            logger.info(f"[{username}] {len(ids_to_delete)}/{len(candidates)} faits retenus pour suppression: {ids_to_delete}")

            if not ids_to_delete:
                logger.info(f"[{username}] Aucun fait ne correspond à la suppression: {query!r}")
                await nexus.publish(delete_results_topic, {"query": query, "deleted_count": 0, "deleted": []})
                return

            # 4. Delete only the filtered facts, collecting associated session_ids
            session_ids_to_delete = set()
            async with aiohttp.ClientSession(headers={"Cookie": f"vk_session={_user_passwords[username]}"}) as http:
                for fact_id in ids_to_delete:
                    try:
                        resp = await http.delete(f"{MNEMONIC_URL}/users/{username}/facts/{fact_id}")
                        resp.raise_for_status()
                        fact = next((f for f in candidates if f["id"] == fact_id), {})
                        label = f"{fact.get('type')}: {fact.get('value')}"
                        deleted_labels.append(label)
                        if fact.get("session_id"):
                            session_ids_to_delete.add(fact["session_id"])
                        logger.info(f"[{username}] Fait supprimé: {fact_id} ({label})")
                    except Exception as e:
                        logger.error(f"[{username}] Échec suppression {fact_id}: {e}")

            # 5. Delete associated sessions
            if session_ids_to_delete:
                async with aiohttp.ClientSession(headers={"Cookie": f"vk_session={_user_passwords[username]}"}) as http:
                    for mnemonic_sid in session_ids_to_delete:
                        try:
                            resp = await http.delete(f"{MNEMONIC_URL}/users/{username}/sessions/{mnemonic_sid}")
                            resp.raise_for_status()
                            logger.info(f"[{username}] Session supprimée: {mnemonic_sid}")
                        except Exception as e:
                            logger.error(f"[{username}] Échec suppression session {mnemonic_sid}: {e}")

        await nexus.publish(
            delete_results_topic,
            {"query": query or str(ids), "deleted_count": len(deleted_labels), "deleted": deleted_labels},
        )
        logger.info(f"[{username}] Résultat suppression publié: {len(deleted_labels)} faits supprimés")

    nexus.subscribe(search_topic, on_search_request)
    nexus.subscribe(delete_topic, on_delete_request)
    nexus.start_listening()
    logger.info(f"[{username}/{session_id}] Abonné à search_preference, delete_facts")


async def main():
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)
    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.start_listening()
    logger.info("Profiler démarré — écoute common/user_connected")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
