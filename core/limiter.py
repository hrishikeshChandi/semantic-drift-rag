from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

print("Setting up the rate limiter...")
limiter = Limiter(key_func=get_remote_address)
print("Rate limited configured with key function: get_remote_address")
