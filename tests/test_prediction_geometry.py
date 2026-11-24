import json

from shapely.geometry import shape

from segmenteer.io.predictions import save_prediction


def test_save_prediction_repairs_self_intersection(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"label": "tissue"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]],
                },
            }
        ],
    }

    path = tmp_path / "prediction.geojson"
    save_prediction(payload, path)
    geometry = shape(json.loads(path.read_text())["features"][0]["geometry"])

    assert geometry.is_valid


def test_save_prediction_preserves_valid_feature_metadata(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"label": "tissue"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]],
                },
            }
        ],
    }

    path = tmp_path / "prediction.geojson"
    save_prediction(payload, path)
    saved_feature = json.loads(path.read_text())["features"][0]

    assert saved_feature["properties"] == payload["features"][0]["properties"]
    assert saved_feature["geometry"] == payload["features"][0]["geometry"]
