def cached_property(f):
    """returns a cached property that is calculated by function f"""
    # From http://code.activestate.com/recipes/576563-cached-property/#:~:text=Cached%20Property%20(Python%20recipe),the%20cached%20value%20is%20returned.

    def get(self):
        try:
            return self._property_cache[f]
        except AttributeError:
            self._property_cache = {}
            x = self._property_cache[f] = f(self)
            return x
        except KeyError:
            x = self._property_cache[f] = f(self)
            return x

    return property(get)