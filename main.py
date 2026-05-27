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

FACT_TYPES = [
    "name", "location", "occupation", "family", "language", "skill",
    "cuisine", "music", "sport", "video_game", "technology", "politics",
    "cinema", "book", "travel", "art", "fashion", "nature", "science",
    "philosophy", "humor", "habit", "goal", "personality", "value",
]

EXTRACT_TOOL = [{
    "type": "function",
    "function": {
        "name": "extract_user_facts",
        "description": (
            "Extraire les faits personnels sur l'utilisateur humain depuis la conversation. "
            "Ne retourner que des faits explicitement mentionnés par l'utilisateur, pas l'assistant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": FACT_TYPES},
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


def _find_topic(private_topics: list, suffix: str) -> str | None:
    for agent_entry in private_topics:
        for t in agent_entry.get("topics", []):
            if t["topic"].endswith(f"/{suffix}"):
                return t["topic"]
    return None


EXTRACT_SYSTEM_PROMPT = (
    "You extract personal facts and interests about the human user from a single message. "
    "Every question or request reveals an interest: asking for weather in Paris → \"is interested in weather in Paris\". "
    "Values must be complete English statements, never French. "
    "Examples: "
    "- \"j aime le retro gaming\" → {\"type\": \"video_game\", \"value\": \"likes retro gaming\"} "
    "- \"parle moi de l architecture de la sega saturn\" → {\"type\": \"technology\", \"value\": \"is interested in Sega Saturn architecture\"} "
    "- \"donne moi la meteo de Paris\" → {\"type\": \"location\", \"value\": \"is interested in weather in Paris\"} "
    "Call extract_user_facts with all facts found in the message."
)


def _extract_facts_for_message_sync(user_message: str) -> list[dict]:
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            tools=EXTRACT_TOOL,
            tool_choice="required",
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return []
        return json.loads(tool_calls[0].function.arguments).get("facts", [])
    except Exception as e:
        logger.error(f"Extraction de faits échouée pour message: {e}")
        return []


async def _extract_facts(messages: list) -> list[dict]:
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    logger.info(f"LLM POST {LLM_BASE_URL}/chat/completions — model={LLM_MODEL}, {len(user_messages)} messages utilisateur")
    logger.info(f"System prompt: {EXTRACT_SYSTEM_PROMPT}")
    all_facts = []
    loop = asyncio.get_event_loop()
    for msg in user_messages:
        logger.info(f"Analyse message: {msg}")
        facts = await loop.run_in_executor(None, _extract_facts_for_message_sync, msg)
        logger.info(f"Faits extraits: {json.dumps(facts, ensure_ascii=False)}")
        all_facts.extend(facts)
    return all_facts


CONSOLIDATE_TOOL = [{
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
                            "type": {"type": "string", "enum": FACT_TYPES},
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


def _find_habits_sync(facts: list[dict]) -> list[dict]:
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
            tools=CONSOLIDATE_TOOL,
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
    if len(all_facts) < HABIT_THRESHOLD:
        logger.info(f"[{username}] Pas assez de faits pour consolidation")
        return

    loop = asyncio.get_event_loop()
    habit_groups = await loop.run_in_executor(None, _find_habits_sync, all_facts)

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


async def on_discussion(username: str, topic: str, payload, user_api_key: str):
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

    logger.info(f"[{username}] Extraction des faits en cours...")
    facts = await _extract_facts(payload)
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

    if username in _subscribed_users:
        logger.debug(f"Utilisateur {username} déjà abonné, skip")
        return
    _subscribed_users.add(username)

    logger.info(f"Nouvel utilisateur: {username} — discussions={discussions_topic}")

    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    profile_topic = f"users/{username}/profile"
    await nexus.publish(
        agent_topics_topic,
        [{
            "agent": AGENT_NAME,
            "topics": [{
                "topic": profile_topic,
                "description": "Profil utilisateur",
                "access": "read",
                "format": {
                    "username": "string",
                    "preferences": {},
                    "history_summary": "string",
                },
            }],
        }],
    )

    await nexus.publish(
        profile_topic,
        {"username": username, "preferences": {}, "history_summary": ""},
        retain=True,
    )

    async def handler(t, p):
        await on_discussion(username, t, p, password)

    nexus.subscribe(discussions_topic, handler)
    nexus.start_listening()
    logger.info(f"Abonné aux discussions de {username}")


async def main():
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)
    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.start_listening()
    logger.info("Profiler démarré — écoute common/user_connected")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
