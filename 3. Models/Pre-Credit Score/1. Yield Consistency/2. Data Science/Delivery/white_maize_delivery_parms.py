config = {
    "ABS_PATH": r"C:\Users\mario\OneDrive\Documents\Work\Clients\Agnify\\",
    "MODEL_PATH": r"C:\Users\mario\OneDrive\Documents\Work\Clients\Agnify\3. Models\Pre-Credit Score\1. Yield Consistency\2. Data Science\Delivery",
    "DATA_FILE_PATH": r"C:\Users\mario\OneDrive\Documents\Work\Clients\Agnify\2. Data\SAFEX\Crop Deliveries",
    "DATA_FILE": "safex_crop_deliveries.parquet",
    "TARGET_VARIABLE": "Crop Delivery (Tonnes)",
    "CROP": "White Maize",
    "PERIOD": 52,
    "ROLLING_AVERAGE_WINDOW": 4,
    "SMOOTHED_TARGET_VARIABLE": "4-Weekly Rolling Average",
    "MODEL_PARMS": {"order": (1, 0, 0), "seasonal_order": (0, 1, 1, 52)},
    "MODEL_FILE": "white_maize_model",
    "TRAIN_SPLIT": 0.8,
}
