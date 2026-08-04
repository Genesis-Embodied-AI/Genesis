import random


def random_scene():
    return random.choice(["🏟️", "🏯", "🏰", "🌃", "🌇", "🌉", "🎡", "⛰️", "🏖️", "🏜️", "🏘️"])


def get_clock(t, speed=10) -> str:
    return "🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛"[int(t * speed) % 12]
