from neo4j import Driver, GraphDatabase

from . import config

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        _driver.verify_connectivity()
    return _driver


def run(query: str, **params):
    with get_driver().session() as session:
        return list(session.run(query, **params))


def close() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
