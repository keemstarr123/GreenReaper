import os
import xarray as xr
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import math
import time
import warnings

import pystac_client
from planetary_computer import sign
from odc.stac import load as stac_load

import rasterio.errors
import pystac_client.exceptions

warnings.filterwarnings("ignore", category=UserWarning)


def create_bounding_box(lat_center, lon_center, half_side_km=2.5):
    lat_delta = half_side_km / 111.32
    lon_delta = half_side_km / (111.32 * math.cos(math.radians(lat_center)))
    return (
        lon_center - lon_delta,
        lat_center - lat_delta,
        lon_center + lon_delta,
        lat_center + lat_delta
    )


def load_data(bounds):
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search_s2 = stac.search(
        bbox=bounds,
        datetime="2020-01-01/2023-12-31",
        collections=['sentinel-2-l2a'],
        query={"eo:cloud_cover": {"lt": 10}}
    )

    search_ls8 = stac.search(
        bbox=bounds,
        datetime="2020-01-01/2023-12-31",
        collections=['landsat-c2-l2'],
        query={"eo:cloud_cover": {"lt": 10}, "platform": {"in": ["landsat-8"]}}
    )

    resolution_m = 10
    scale_deg = resolution_m / 111320.0

    ds_s2 = load_s2(search_s2, scale_deg, bounds)
    ds_ls8 = load_ls8(search_ls8, scale_deg, bounds)
    return ds_s2, ds_ls8


def load_s2(search, scale_deg, bounds):
    items = list(search.items())
    if not items:
        print("No Sentinel-2 items found.")
        return None
    return stac_load(
        items,
        bands=["B03", "B04", "B08", "B11"],
        crs="EPSG:4326",
        resolution=scale_deg,
        chunks={"x": 512, "y": 512},
        dtype="uint16",
        patch_url=sign,
        bbox=bounds
    )


def load_ls8(search, scale_deg, bounds):
    items = list(search.items())
    if not items:
        print("No Landsat-8 items found.")
        return None
    ds = stac_load(
        items,
        bands=["lwir11"],
        crs="EPSG:4326",
        resolution=scale_deg,
        chunks={"x": 512, "y": 512},
        dtype="uint16",
        patch_url=sign,
        bbox=bounds
    )
    scale2 = 0.00341802
    offset2 = 149.0
    kelvin_to_celsius = 273.15
    return ds.astype(float) * scale2 + offset2 - kelvin_to_celsius


def calculate_median(ds):
    if ds is None:
        return None
    return ds.median(dim="time").compute()


def compute_ndvi(ds):
    return (ds.B08 - ds.B04) / (ds.B08 + ds.B04)


def compute_ndbi(ds):
    return (ds.B11 - ds.B08) / (ds.B11 + ds.B08)


def compute_ndwi(ds):
    return (ds.B03 - ds.B08) / (ds.B03 + ds.B08)


def load_csv_data(csv_path: str, limit: int = 99) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.iloc[:limit]


def extract_features_from_zarr(zarr_directory: str, df_labels_subset: pd.DataFrame):
    X_list, y_list, processed_boxes = [], [], []

    for idx, row in df_labels_subset.iterrows():
        box_id = row["BoxID"]
        uhi_val = row["UHI_Index"]
        zarr_path = os.path.join(zarr_directory, f"satellite_features_box{box_id}.zarr")

        if not os.path.isdir(zarr_path):
            print(f"WARNING: No Zarr folder found for BoxID={box_id}: {zarr_path}")
            continue

        ds = xr.open_zarr(zarr_path)

        try:
            ndvi = ds.NDVI.isel(latitude=slice(0, 200), longitude=slice(0, 200))
            ndbi = ds.NDBI.isel(latitude=slice(0, 200), longitude=slice(0, 200))
            ndwi = ds.NDWI.isel(latitude=slice(0, 200), longitude=slice(0, 200))
            lst = ds.LST.isel(latitude=slice(0, 200), longitude=slice(0, 200))
        except AttributeError:
            print(f"ERROR: Data variables missing in {zarr_path}. Skipping.")
            continue

        features = [
            float(ndvi.mean().values), float(ndbi.mean().values),
            float(ndwi.mean().values), float(lst.mean().values),
            float(ndvi.std().values), float(ndbi.std().values),
            float(ndwi.std().values), float(lst.std().values)
        ]

        X_list.append(features)
        y_list.append(uhi_val)
        processed_boxes.append(box_id)

    return np.array(X_list), np.array(y_list), processed_boxes


def extract_features_from_single_zarr(zarr_path: str):
    if not os.path.isdir(zarr_path):
        print(f"ERROR: Zarr folder not found at {zarr_path}")
        return None

    ds = xr.open_zarr(zarr_path)

    try:
        ndvi = ds.NDVI.isel(latitude=slice(0, 200), longitude=slice(0, 200))
        ndbi = ds.NDBI.isel(latitude=slice(0, 200), longitude=slice(0, 200))
        ndwi = ds.NDWI.isel(latitude=slice(0, 200), longitude=slice(0, 200))
        lst = ds.LST.isel(latitude=slice(0, 200), longitude=slice(0, 200))
    except AttributeError:
        print(f"ERROR: Required variables missing in {zarr_path}")
        return None

    features = [
        float(ndvi.mean().values), float(ndbi.mean().values),
        float(ndwi.mean().values), float(lst.mean().values),
        float(ndvi.std().values), float(ndbi.std().values),
        float(ndwi.std().values), float(lst.std().values)
    ]

    return np.array(features).reshape(1, -1)


def scale_and_split_data(X, y, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE:  {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R^2:  {r2:.3f}")


def main():
    

    user_lat = float(input("Please enter the Latitude : "))
    user_lon = float(input("Please enter the Longitude : "))
    start_time = time.time()  # ⏱️ Start timing

    output_path = "user.zarr"

    print(f"Processing location: lat={user_lat}, lon={user_lon}")
    bounds = create_bounding_box(user_lat, user_lon, half_side_km=2.5)

    try:
        ds_s2, ds_ls8 = load_data(bounds)
        if ds_s2 is None or ds_ls8 is None:
            print("ERROR: Missing data from search.")
            return

        s2_median = calculate_median(ds_s2)
        ls8_median = calculate_median(ds_ls8)
        if s2_median is None or ls8_median is None:
            print("ERROR: No valid median data.")
            return

        ndvi = compute_ndvi(s2_median)
        ndbi = compute_ndbi(s2_median)
        ndwi = compute_ndwi(s2_median)
        lst = ls8_median.lwir11

        ds_combined = xr.Dataset({
            "NDVI": ndvi,
            "NDBI": ndbi,
            "NDWI": ndwi,
            "LST": lst
        })

        ds_combined.to_zarr(output_path, mode="w")
        print(f"✅ Zarr dataset saved to '{output_path}'")

    except Exception as e:
        print(f"❌ Failed to process: {e}")
        return

    zarr_directory = "/Users/vishnuram/Desktop/EY_GLOBALCHAMP-main"
    csv_path = "/Users/vishnuram/Desktop/EY_GLOBALCHAMP-main/26:2/ProcessedTrainingData_real_modified.csv"

    df_subset = load_csv_data(csv_path, limit=99)
    X, y, processed_boxes = extract_features_from_zarr(zarr_directory, df_subset)

    X_train, X_test, y_train, y_test, scaler = scale_and_split_data(X, y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    print("\n🔍 Predicting UHI Index for user.zarr...")
    user_features = extract_features_from_single_zarr(output_path)
    if user_features is None:
        print("❌ Failed to extract features for prediction.")
        return

    user_features_scaled = scaler.transform(user_features)
    predicted_uhi = model.predict(user_features_scaled)[0]

    print(f"🌡️  Predicted UHI Index at ({user_lat}, {user_lon}): {predicted_uhi:.4f}")

    end_time = time.time()  # ⏱️ End timing
    duration = end_time - start_time
    print(f"\n⏳ Total execution time: {duration:.2f} seconds")
if __name__ == "__main__":
    main()
