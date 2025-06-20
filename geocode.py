import os
import time
import logging
import pandas as pd
import requests
from pymongo import MongoClient, errors
from tqdm import tqdm

# ===== CONFIG =====
API_KEY = "YOUR_GOOGLE_GEOCODING_API_KEY"
MONGO_URI = "mongodb+srv://vishnu:vE83Oq559WV0pcLg@city-state.morcjqg.mongodb.net/"
MONGO_DB = "City_State_pincode"
MONGO_COLLECTION = "wockhard_5k"
OUTPUT_FOLDER = "city_state_pincode"
INPUT_FOLDER = "cleaned"

# ===== SETUP LOGGING =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===== ENSURE FOLDERS EXIST =====
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===== CONNECT TO MONGODB =====
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
except errors.ServerSelectionTimeoutError as err:
    logging.critical(f"❌ Failed to connect to MongoDB: {err}")
    raise SystemExit(1)


def get_location_details(record_id, address):
    endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": API_KEY}
    max_retries = 3
    attempts = 0

    while attempts < max_retries:
        try:
            response = requests.get(endpoint, params=params, timeout=10)

            if response.status_code != 200:
                logging.warning(f"🔁 Attempt {attempts + 1}: Bad response {response.status_code} for address: {address}")
                attempts += 1
                time.sleep(1)
                continue

            data = response.json()

            if data.get('status') != 'OK':
                logging.warning(f"🛑 Geocoding failed for address '{address}' with status: {data.get('status')}")
                return None

            results = data['results'][0]
            location = results['geometry']['location']
            latitude = location.get('lat')
            longitude = location.get('lng')
            city = state = pincode = area = None

            for component in results.get('address_components', []):
                if "locality" in component['types']:
                    city = component.get('long_name')
                elif "administrative_area_level_1" in component['types']:
                    state = component.get('long_name')
                elif "postal_code" in component['types']:
                    pincode = component.get('long_name')
                elif "sublocality" in component['types']:
                    area = component.get('long_name')

            return {
                "record_id": record_id,
                "address": address,
                "city": city,
                "state": state,
                "pincode": pincode,
                "area": area,
                "latitude": latitude,
                "longitude": longitude
            }

        except requests.exceptions.Timeout:
            logging.warning(f"⏰ Timeout on attempt {attempts + 1} for address '{address}'")
        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Request exception on attempt {attempts + 1} for address '{address}': {e}")
        except Exception as e:
            logging.exception(f"💥 Unexpected error during geocoding for address '{address}': {e}")

        attempts += 1
        time.sleep(1)

    return None


def geocode_csv_file(filename, uid_col='UID Multiplier', address_col='Multiplier Address'):
    input_path = os.path.join(INPUT_FOLDER, 'NMC.gsk_new4k_qc_nm_st_merged_names_cleaned_predicted_1.csv')

    if not os.path.exists(input_path):
        logging.error(f"📂 Input file not found: {input_path}")
        return

    try:
        data = pd.read_csv(input_path)
    except Exception as e:
        logging.critical(f"❌ Failed to read CSV file: {e}")
        return

    if uid_col not in data.columns or address_col not in data.columns:
        logging.error(f"🚫 Missing required columns '{uid_col}' or '{address_col}' in the input file.")
        return

    logging.info(f"🔍 Starting geocoding for {len(data)} rows...")

    for i, row in tqdm(data.iterrows(), total=len(data), desc="📍 Geocoding", unit="row"):
        try:
            address = str(row[address_col]).strip()
            record_id = row[uid_col]

            if not address or address.lower() in ['nan', 'n/a', 'na', '#n/a']:
                logging.info(f"⏭️ Skipping row {i} due to invalid address: {address}")
                continue

            details = get_location_details(record_id, address)

            if details:
                try:
                    collection.update_one(
                        {"record_id": details["record_id"]},
                        {"$set": details},
                        upsert=True
                    )
                    time.sleep(0.1)
                except Exception as db_err:
                    logging.error(f"❌ MongoDB error on row {i}: {db_err}")
            else:
                logging.warning(f"⚠️ Geocoding failed for address: {address}")

        except Exception as e:
            logging.exception(f"💥 Unexpected error processing row {i}: {e}")

    # Export to CSV
    try:
        documents = list(collection.find({}, {"_id": 0}))
        if documents:
            output_path = os.path.join(OUTPUT_FOLDER, "geocoded_results.csv")
            pd.DataFrame(documents).to_csv(output_path, index=False)
            logging.info(f"✅ Exported geocoded data to: {output_path}")
        else:
            logging.warning("⚠️ No data found in MongoDB to export.")
    except Exception as e:
        logging.critical(f"❌ Failed to export data to CSV: {e}")
