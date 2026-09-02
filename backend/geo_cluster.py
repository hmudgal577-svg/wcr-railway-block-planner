"""
geo_cluster.py
---------------
Geospatial multi-department bundling layer.

Uses GeoPandas + Shapely (pure open-source geometry stack) to find maintenance requests
raised by DIFFERENT departments within radius_m (500m).

Resource Constraint Matrix:
- If a task has `is_heavy_machinery` / `exclusive_block` == True (e.g. TRT train / heavy tamping),
  it is flagged as EXCLUSIVE and strictly bypassed from multi-department bundling for site
  safety & logistical compliance.
"""

from typing import List
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEOPANDAS_AVAILABLE = False


def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_bundling_clusters(requests_df: pd.DataFrame, radius_m: float = 500.0) -> pd.DataFrame:
    """
    Returns a copy of requests_df with an added `bundle_cluster` column.
    A value of -1 means "not bundled" (lone or exclusive job).
    Positive integers group 2+ compatible cross-department requests within radius_m.
    """
    out = requests_df.copy()
    out["bundle_cluster"] = -1

    if len(out) < 2:
        return out

    # Determine exclusive tasks that must bypass bundling
    is_exclusive = np.zeros(len(out), dtype=bool)
    if "is_heavy_machinery" in out.columns:
        is_exclusive = is_exclusive | out["is_heavy_machinery"].fillna(False).astype(bool).values
    if "exclusive_block" in out.columns:
        is_exclusive = is_exclusive | out["exclusive_block"].fillna(False).astype(bool).values

    if not GEOPANDAS_AVAILABLE:
        return _fallback_haversine_clusters(out, radius_m, is_exclusive)

    centroid_lon = out["longitude"].mean()
    centroid_lat = out["latitude"].mean()
    epsg = _utm_epsg_for_lonlat(centroid_lon, centroid_lat)

    gdf = gpd.GeoDataFrame(
        out,
        geometry=[Point(xy) for xy in zip(out["longitude"], out["latitude"])],
        crs="EPSG:4326",
    ).to_crs(epsg=epsg)

    n = len(gdf)
    uf = _UnionFind(n)
    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    depts = out["department"].values
    corridors = out["corridor"].values

    for i in range(n):
        if is_exclusive[i]:
            continue  # Bypassed: exclusive heavy machinery
        for j in range(i + 1, n):
            if is_exclusive[j]:
                continue
            if depts[i] == depts[j]:
                continue  # only cross-department pairs count as bundling
            if corridors[i] != corridors[j]:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= radius_m:
                uf.union(i, j)

    out["bundle_cluster"] = _assign_cluster_ids(uf, n, is_exclusive)
    return out


def _assign_cluster_ids(uf: "_UnionFind", n: int, is_exclusive: np.ndarray) -> List[int]:
    cluster_members = {}
    for i in range(n):
        if is_exclusive[i]:
            continue
        root = uf.find(i)
        cluster_members.setdefault(root, []).append(i)

    cluster_id = 0
    labels = [-1] * n
    for root, members in cluster_members.items():
        if len(members) < 2:
            continue
        for m in members:
            labels[m] = cluster_id
        cluster_id += 1
    return labels


def _fallback_haversine_clusters(out: pd.DataFrame, radius_m: float, is_exclusive: np.ndarray) -> pd.DataFrame:
    R = 6371000.0
    lat = np.radians(out["latitude"].values)
    lon = np.radians(out["longitude"].values)
    depts = out["department"].values
    corridors = out["corridor"].values
    n = len(out)
    uf = _UnionFind(n)

    for i in range(n):
        if is_exclusive[i]:
            continue
        for j in range(i + 1, n):
            if is_exclusive[j]:
                continue
            if depts[i] == depts[j] or corridors[i] != corridors[j]:
                continue
            dlat = lat[j] - lat[i]
            dlon = lon[j] - lon[i]
            a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat[j]) * np.sin(dlon / 2) ** 2
            dist = 2 * R * np.arcsin(np.sqrt(a))
            if dist <= radius_m:
                uf.union(i, j)

    out["bundle_cluster"] = _assign_cluster_ids(uf, n, is_exclusive)
    return out
