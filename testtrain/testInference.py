from inference import predict_snr, WINDOW_SIZE

# dummy input (HARUS sesuai format)
dummy_sequence = []

for _ in range(WINDOW_SIZE):
    dummy_sequence.append({
        "temperature_C": 30,
        "humidity_percent": 70,
        "rain_rate_mmhr": 50,
        "wind_speed_ms": 2,
        "pressure_hPa": 1010,
        "hour_of_day": 12,
        "day_of_week": 3,
        "month": 6
    })

# run prediction
result = predict_snr(dummy_sequence)

print("Predicted SNR:", result)