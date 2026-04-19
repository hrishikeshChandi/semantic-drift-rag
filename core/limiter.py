from core.logging_config import get_logger

logger = get_logger(__name__)


from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

logger.info("Setting up the rate limiter...")
limiter = Limiter(key_func=get_remote_address)
logger.info("Rate limited configured with key function: get_remote_address")
