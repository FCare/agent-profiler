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
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY = os.environ.get("LLAMACPP_API_KEY", "no-key")

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


def _extract_facts_sync(messages: list) -> list[dict]:
    try:
        client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze conversations to extract personal facts about the human user (not the assistant). "
                        "Only return facts explicitly mentioned in the conversation. "
                        "Always write fact values in English, regardless of the conversation language."
                    ),
                },
                *messages,
            ],
            tools=EXTRACT_TOOL,
            tool_choice={"type": "function", "function": {"name": "extract_user_facts"}},
        )
        args = resp.choices[0].message.tool_calls[0].function.arguments
        return json.loads(args).get("facts", [])
    except Exception as e:
        logger.error(f"Extraction de faits échouée: {e}")
        return []


async def _extract_facts(messages: list) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_facts_sync, messages)


async def on_discussion(username: str, topic: str, payload, user_api_key: str):
    if not isinstance(payload, list) or not payload:
        return

    logger.info(f"[{username}] Discussion reçue ({len(payload)} messages)")

    auth_headers = {"X-API-Key": user_api_key}

    try:
        async with aiohttp.ClientSession(headers=auth_headers) as http:
            resp = await http.post(
                f"{MNEMONIC_URL}/users/{username}/sessions",
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
