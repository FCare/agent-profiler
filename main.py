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

AGENT_NAME = "profiler"

_subscribed_users: set[str] = set()

DEFAULT_FACT_TYPES = [
    "name", "location", "occupation", "family", "language", "skill",
    "cuisine", "music", "sport", "video_game", "technology", "politics",
    "cinema", "book", "travel", "art", "fashion", "nature", "science",
    "philosophy", "humor", "habit", "goal", "personality",
]


def _type_field_schema(known_types: list[str]) -> dict:
    return {
        "type": "string",
        "description": (
            f"Fact category. Prefer one of the known types if it fits: {', '.join(known_types)}. "
            "Otherwise invent a concise English noun (e.g. 'sport', 'cinema')."
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
    "Examples:\n"
    "- [user]: quelle meteo demain a paris → {type: \"location\", value: \"is interested in weather in Paris\"}\n"
    "- [user]: j aime le retro gaming → {type: \"video_game\", value: \"likes retro gaming\"}\n"
    "- [user]: tu connais le cycle de Hain? / [assistant]: c'est une série SF / [user]: c'est plusieurs livres → {type: \"book\", value: \"is interested in the Hain cycle\"}\n"
    "- [user]: j'aime le space opera → {type: \"book\", value: \"enjoys space opera as a genre\"}\n"
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

    logger.info(f"[{username}] {len(all_facts)} faits récupérés (seuil: {HABIT_THRESHOLD})")

    type_counts = {}
    for f in all_facts:
        type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1
    logger.info(f"[{username}] Répartition par type: {type_counts}")

    candidate_facts = [f for f in all_facts if type_counts[f["type"]] >= HABIT_THRESHOLD]
    if not candidate_facts:
        logger.info(f"[{username}] Aucun type avec ≥{HABIT_THRESHOLD} faits, pas de consolidation")
        return
    logger.info(f"[{username}] {len(candidate_facts)} faits candidats (types: {[t for t, c in type_counts.items() if c >= HABIT_THRESHOLD]})")

    known_types = sorted(set(f["type"] for f in all_facts)) or DEFAULT_FACT_TYPES
    loop = asyncio.get_event_loop()
    habit_groups = await loop.run_in_executor(None, _find_habits_sync, candidate_facts, known_types)

    if not habit_groups:
        logger.info(f"[{username}] Aucune habitude détectée")
        return

    logger.info(f"[{username}] {len(habit_groups)} habitude(s) détectée(s)")
    facts_by_id = {f["id"]: f for f in all_facts}

    for habit in habit_groups:
        valid_ids = [fid for fid in habit["fact_ids"] if fid in facts_by_id]
        logger.info(f"[{username}] Habitude '{habit['description']}': {len(valid_ids)} faits valides sur {len(habit['fact_ids'])} proposés")
        if len(valid_ids) < HABIT_THRESHOLD:
            logger.info(f"[{username}] Ignoré — moins de {HABIT_THRESHOLD} faits valides")
            continue

        session_ids = list(dict.fromkeys(facts_by_id[fid]["session_id"] for fid in valid_ids))
        logger.info(f"[{username}] Stockage habitude: type={habit['type']} description=\"{habit['description']}\" sessions={session_ids}")

        try:
            async with aiohttp.ClientSession(headers=auth_headers) as http:
                resp = await http.post(
                    f"{MNEMONIC_URL}/users/{username}/facts",
                    json={
                        "facts": [{"type": habit["type"], "value": habit["description"]}],
                        "session_id": session_ids[0],
                        "session_ids": session_ids,
                        "is_habit": True,
                    },
                )
                resp.raise_for_status()
                logger.info(f"[{username}] Habitude stockée dans mnemonic")

                for fid in valid_ids:
                    del_resp = await http.delete(f"{MNEMONIC_URL}/users/{username}/facts/{fid}")
                    logger.info(f"[{username}] Fait {fid} supprimé (status {del_resp.status})")

            logger.info(f"[{username}] Consolidation terminée: {len(valid_ids)} faits → 1 habitude")
        except Exception as e:
            logger.error(f"[{username}] Échec consolidation habitude: {e}")


SELECT_TYPES_TOOL = [{
    "type": "function",
    "function": {
        "name": "select_profile_types",
        "description": "Select fact types that describe core personal attributes of the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types that describe who the person IS (name, location, occupation, family, language, personality, goal, skill). Exclude pure hobby/interest types (book, music, video_game, cinema) as those are covered by habits.",
                }
            },
            "required": ["types"],
        },
    },
}]


def _select_profile_types_sync(available_types: list[str]) -> list[str]:
    if not available_types:
        return []
    logger.info(f"Sélection des types de profil parmi: {available_types}")
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Select the fact types from the provided list that describe core personal attributes (name, location, occupation, family, language, personality, goal, skill). "
                    "Exclude pure hobby/interest types (book, music, video_game, cinema, sport, food) — those are covered by habits. "
                    "Call select_profile_types with the relevant types."
                )},
                {"role": "user", "content": f"Available types: {', '.join(available_types)}"},
            ],
            tools=SELECT_TYPES_TOOL,
            tool_choice="required",
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return []
        result = json.loads(tool_calls[0].function.arguments).get("types", [])
        logger.info(f"Types de profil sélectionnés: {result}")
        return result
    except Exception as e:
        logger.error(f"Sélection des types de profil échouée: {e}")
        return []


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
                            "Only the 1 or 2 types whose facts would directly name the subject. "
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
                    "Call select_types with at most 2 types."
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
        lines.append("Personal facts:")
        for f in personal_facts:
            lines.append(f"  [{f['type']}] {f['value']}")
    if habits:
        lines.append("Habits and recurring interests:")
        for h in habits:
            lines.append(f"  [{h['type']}] {h['value']}")
    context = "\n".join(lines)
    logger.info(f"[{username}] Contexte profil envoyé au LLM:\n{context}")
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Generate a concise personal profile of the user from the provided facts and habits. "
                    "Write in third person, in English. "
                    "Start with personal info (name, location, occupation, family if known), then describe recurring habits and interests. "
                    "Be factual, concise, and natural-sounding. 3-6 sentences."
                )},
                {"role": "user", "content": f"User: {username}\n\n{context}"},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[{username}] Génération du texte de profil échouée: {e}")
        return ""


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
    try:
        async with aiohttp.ClientSession(headers=auth_headers) as http:
            resp = await http.get(f"{MNEMONIC_URL}/users/{username}/facts/types")
            resp.raise_for_status()
            available_types = (await resp.json()).get("types", [])
    except Exception as e:
        logger.error(f"[{username}] Échec récupération types pour profil: {e}")
        return

    logger.info(f"[{username}] Types disponibles: {available_types}")
    if not available_types:
        logger.info(f"[{username}] Aucun type disponible, profil ignoré")
        return

    loop = asyncio.get_event_loop()
    profile_types = await loop.run_in_executor(None, _select_profile_types_sync, available_types)
    logger.info(f"[{username}] Types retenus pour faits personnels: {profile_types}")

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


async def on_discussion(username: str, topic: str, payload, user_api_key: str, nexus, profile_topic: str):
    if not isinstance(payload, list) or not payload:
        return

    logger.info(f"[{username}] Discussion reçue ({len(payload)} messages)")

    auth_headers = {"Cookie": f"vk_session={user_api_key}"}
    sessions_url = f"{MNEMONIC_URL}/users/{username}/sessions"
    logger.info(f"[{username}] POST {sessions_url} — Cookie: vk_session={user_api_key}")

    try:
        async with aiohttp.ClientSession(headers=auth_headers) as http:
            resp = await http.post(
                sessions_url,
                json={"messages": payload},
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
    facts = await _extract_facts(payload, known_types)
    if not facts:
        logger.info(f"[{username}] Aucun fait extrait")
        return

    logger.info(f"[{username}] {len(facts)} faits extraits:")
    for fact in facts:
        logger.info(f"[{username}]   {fact['type']}: {fact['value']}")

    try:
        async with aiohttp.ClientSession(headers=auth_headers) as http:
            resp = await http.post(
                f"{MNEMONIC_URL}/users/{username}/facts",
                json={"facts": facts, "session_id": session_id},
            )
            resp.raise_for_status()
        logger.info(f"[{username}] Faits enregistrés dans mnemonic")
    except Exception as e:
        logger.error(f"[{username}] Échec enregistrement des faits dans mnemonic: {e}")
        return

    await _consolidate_habits(username, auth_headers)
    await _generate_profile(username, auth_headers, nexus, profile_topic)


async def on_user_connected(topic: str, payload):
    if not isinstance(payload, dict):
        return

    username = payload.get("username")
    password = payload.get("password")
    private_topics = payload.get("private_topics", [])

    if not username or not password:
        return

    discussions_topic = _find_topic(private_topics, "discussions")
    agent_topics_topic = _find_topic(private_topics, "agent_topics")

    if not discussions_topic or not agent_topics_topic:
        logger.warning(f"Topics manquants pour {username}, skip")
        return

    already_subscribed = username in _subscribed_users

    profile_topic = f"users/{username}/profile"
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    search_topic = f"users/{username}/search_preference"
    delete_topic = f"users/{username}/delete_facts"
    search_results_topic = f"users/{username}/search_results"
    delete_results_topic = f"users/{username}/delete_results"

    # Always republish topic declaration so agents that restarted can rediscover it
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
                    "description": "Recherche sémantique de faits utilisateur",
                    "access": "write",
                    "response_topic": search_results_topic,
                    "format": {"query": "string", "n": 5},
                },
                {
                    "topic": delete_topic,
                    "description": (
                        "Supprime des faits mémorisés. "
                        "UNIQUEMENT si l'utilisateur demande EXPLICITEMENT de supprimer, effacer ou oublier quelque chose "
                        "(ex: 'supprime', 'efface', 'oublie', 'retire de ta mémoire'). "
                        "NE PAS utiliser si l'utilisateur dit simplement qu'il ne s'intéresse plus à quelque chose "
                        "ou exprime une préférence changeante sans demander explicitement la suppression."
                    ),
                    "access": "write",
                    "response_topic": delete_results_topic,
                    "format": {"query": "string (OR) ids: [\"...\"]"},
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
    )
    logger.info(f"[{username}] Topics déclarés sur {agent_topics_topic}")

    if already_subscribed:
        logger.debug(f"[{username}] Déjà abonné aux discussions, skip souscription")
        return

    _subscribed_users.add(username)
    logger.info(f"Nouvel utilisateur: {username} — discussions={discussions_topic}")

    auth_headers = {"Cookie": f"vk_session={password}"}

    async def handler(t, p):
        await on_discussion(username, t, p, password, nexus, profile_topic)

    async def on_search_request(t, p):
        if not isinstance(p, dict):
            return
        query = p.get("query", "")
        n = int(p.get("n", 5))
        if not query:
            return
        logger.info(f"[{username}] Recherche de faits: {query!r} (n={n})")

        loop = asyncio.get_event_loop()

        # 1. Fetch available types
        available_types = await _fetch_known_types(username, auth_headers)
        logger.info(f"[{username}] Types disponibles: {available_types}")

        # 2. LLM selects relevant types for this query
        selected_types = await loop.run_in_executor(
            None, _select_search_types_sync, query, available_types
        )
        logger.info(f"[{username}] Types retenus pour la recherche: {selected_types}")

        # 3. Fetch facts by type; fall back to semantic search if no types matched
        facts = []
        if selected_types:
            try:
                async with aiohttp.ClientSession(headers=auth_headers) as http:
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
                async with aiohttp.ClientSession(headers=auth_headers) as http:
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
        ids = p.get("ids")
        query = p.get("query", "")
        if not ids and not query:
            return

        deleted_labels = []

        if ids:
            logger.info(f"[{username}] Suppression par ids: {ids}")
            async with aiohttp.ClientSession(headers=auth_headers) as http:
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
            available_types = await _fetch_known_types(username, auth_headers)
            selected_types = await loop.run_in_executor(
                None, _select_deletion_types_sync, query, available_types
            )
            logger.info(f"[{username}] Types retenus pour suppression: {selected_types}")

            # 2. Fetch candidate facts by type
            candidates = []
            if selected_types:
                try:
                    async with aiohttp.ClientSession(headers=auth_headers) as http:
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

            # 4. Delete only the filtered facts
            async with aiohttp.ClientSession(headers=auth_headers) as http:
                for fact_id in ids_to_delete:
                    try:
                        resp = await http.delete(f"{MNEMONIC_URL}/users/{username}/facts/{fact_id}")
                        resp.raise_for_status()
                        fact = next((f for f in candidates if f["id"] == fact_id), {})
                        label = f"{fact.get('type')}: {fact.get('value')}"
                        deleted_labels.append(label)
                        logger.info(f"[{username}] Fait supprimé: {fact_id} ({label})")
                    except Exception as e:
                        logger.error(f"[{username}] Échec suppression {fact_id}: {e}")

        await nexus.publish(
            delete_results_topic,
            {"query": query or str(ids), "deleted_count": len(deleted_labels), "deleted": deleted_labels},
        )
        logger.info(f"[{username}] Résultat suppression publié: {len(deleted_labels)} faits supprimés")

    nexus.subscribe(discussions_topic, handler)
    nexus.subscribe(search_topic, on_search_request)
    nexus.subscribe(delete_topic, on_delete_request)
    nexus.start_listening()
    logger.info(f"[{username}] Abonné aux discussions, search_preference, delete_facts")


async def main():
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)
    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.start_listening()
    logger.info("Profiler démarré — écoute common/user_connected")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
