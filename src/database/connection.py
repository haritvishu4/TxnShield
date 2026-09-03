from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.database.models import Base
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("database")

_engine = None
_SessionFactory = None

def init_db(database_url: str = None) -> None:
    """Initializes SQLite database engine and creates tables if not existing."""
    global _engine, _SessionFactory
    if database_url is None:
        cfg = load_config()
        database_url = cfg["paths"]["database_url"]

    logger.info(f"Initializing database with URL: {database_url}")
    _engine = create_engine(database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=_engine)
    _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    logger.info("Database initialized and schema verified.")

def get_db_session() -> Generator[Session, None, None]:
    """Provides a transactional database session scope."""
    global _SessionFactory
    if _SessionFactory is None:
        init_db()
    session = _SessionFactory()
    try:
        yield session
    finally:
        session.close()

def get_session_direct() -> Session:
    """Returns a new session directly for script/dashboard usage."""
    global _SessionFactory
    if _SessionFactory is None:
        init_db()
    return _SessionFactory()
