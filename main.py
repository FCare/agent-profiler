import asyncio
import json
import logging
import os
import sys

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

AGENT_NAME = "profiler"

MY_TOPICS = [
    {
        "topic": None,  # filled per-user
        "description": "Profil utilisateur",
        "access": "read",
        "format": {
            "username": "string",
            "preferences": {},
            "history_summary": "string",
        },
    }
]


def _find_topic(private_topics: list, suffix: str) -> str | None:
    for agent_entry in private_topics:
        for t in agent_entry.get("topics", []):
            if t["topic"].endswith(f"/{suffix}"):
                return t["topic"]
    return None


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

    logger.info(f"Nouvel utilisateur: {username} — discussions={discussions_topic}, agent_topics={agent_topics_topic}")

    # Connexion en tant que service
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    # Déclarer le topic profile sur agent_topics
    profile_topic = f"users/{username}/profile"
    await nexus.publish(
        agent_topics_topic,
        [
            {
                "agent": AGENT_NAME,
                "topics": [
                    {
                        "topic": profile_topic,
                        "description": "Profil utilisateur",
                        "access": "read",
                        "format": {
                            "username": "string",
                            "preferences": {},
                            "history_summary": "string",
                        },
                    }
                ],
            }
        ],
    )
    logger.info(f"Topic profile déclaré sur {agent_topics_topic}")

    # Publier un profil initial (retained) si pas encore existant
    await nexus.publish(
        profile_topic,
        {"username": username, "preferences": {}, "history_summary": ""},
        retain=True,
    )
    logger.info(f"Profil initial publié sur {profile_topic} (retained)")

    # S'abonner aux discussions
    nexus.subscribe(discussions_topic, lambda t, p: on_discussion(username, t, p))
    nexus.start_listening()
    logger.info(f"Abonné aux discussions de {username}")


def on_discussion(username: str, topic: str, payload):
    logger.info(f"[{username}] Nouveau message sur {topic}: {json.dumps(payload, ensure_ascii=False)[:120]}")
    # TODO: mettre à jour le profil en fonction de la conversation


async def main():
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)
    nexus.subscribe("common/user_connected", lambda t, p: asyncio.get_event_loop().create_task(on_user_connected(t, p)))
    nexus.start_listening()
    logger.info(f"Profiler démarré — écoute common/user_connected")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
