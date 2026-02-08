# Private module - use gs.engine.entities.* for public API
#
# Note: We don't import states, materials, force_fields here because they're
# imported directly by genesis/__init__.py to avoid recursion issues.
# We only use lazy loading for 'entities' which users access via gs.engine.entities.*


def __getattr__(name):
    if name == "entities":
        # Lazy import entities to allow gs.init() to be called first
        from . import entities

        return entities
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
