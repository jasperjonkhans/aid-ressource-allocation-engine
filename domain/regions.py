from __future__ import annotations

from project.config import CONFIG


_DOMAIN_CONFIG = CONFIG["domain"]

GEDO_POLYGON = [tuple(point) for point in _DOMAIN_CONFIG["gedo_polygon"]]
BURHAKABA_POLYGON = [tuple(point) for point in _DOMAIN_CONFIG["burhakaba_polygon"]]
BAKOOL_POLYGON = [tuple(point) for point in _DOMAIN_CONFIG["bakool_polygon"]]

buur_hakaba_districts = list(_DOMAIN_CONFIG["buur_hakaba_districts"])
bakool_districts = list(_DOMAIN_CONFIG["bakool_districts"])
gedo_districts = list(_DOMAIN_CONFIG["gedo_districts"])

water_region_districts = {
    "buur_hakaba": buur_hakaba_districts,
    "bakool": bakool_districts,
    "gedo": gedo_districts,
}
