from __future__ import annotations

from project.config import CONFIG


_DOMAIN_CONFIG = CONFIG["domain"]

GEDO_POLYGON = [tuple(point) for point in _DOMAIN_CONFIG["gedo_polygon"]]
BURHAKABA_POLYGON = [tuple(point) for point in _DOMAIN_CONFIG["burhakaba_polygon"]]
BAKOOL_POLYGON = [tuple(point) for point in _DOMAIN_CONFIG["bakool_polygon"]]

bay_districts = list(_DOMAIN_CONFIG["bay_districts"])
bakool_districts = list(_DOMAIN_CONFIG["bakool_districts"])
gedo_districts = list(_DOMAIN_CONFIG["gedo_districts"])

water_region_districts = {
    "bay": bay_districts,
    "bakool": bakool_districts,
    "gedo": gedo_districts,
}
