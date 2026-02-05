import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy.stats import nakagami
import os

class SimSim:
    def __init__(self, frequency_hz: float, environtment_mode: str = "lab"):
        self.c = 3e8  # Speed of light in m/s
        self.kb = 1.38064852e-23  # Boltzmann constant in J/K
        self.T0 = 290  # Standard temperature in Kelvin
        
        self.frequency = frequency_hz
        self.frequency_ghz = frequency_hz / 1e9
        self.wavelength_m = self.c / self.frequency
        assert self.frequency_ghz in [140, 220], \
            "hanya frekuensi 140 GHz dan 220 GHz yang didukung"
            
        assert environtment_mode in ["lab", "urban"], \
            "environtment mode harus 'lab' atau 'urban"
        self.environtment_mode = environtment_mode
        self.lab_mode = (environtment_mode == "lab")
        
        if self.lab_mode:
            self.p_block_start = 0.002 / 60
        else:
            self.p_block_start = 0.05 / 60
        
        if self.frequency_ghz == 140:
            self.tx_power_dbm = 20.0
            self.tx_antenna_gain_dbi = 40.0
            self.rx_antenna_gain_dbi = 40.0
            self.rx_noise_figure_db = 12.0
            self.bandwidth_hz = 1e9
            
        else:
            self.tx_power_dbm = 15.0
            self.tx_antenna_gain_dbi = 45.0
            self.rx_antenna_gain_dbi = 45.0
            self.rx_noise_figure_db = 16.5
            self.bandwidth_hz = 2e9
            
        self.eirp_dbm = self.tx_power_dbm + self.tx_antenna_gain_dbi
        
        gain_linear = 10 ** (self.tx_antenna_gain_dbi / 10)
        D_over_lambda = np.sqrt(gain_linear / np.pi)
        self.beamwidth_deg = 70 / D_over_lambda
        
        self.tx_position = np.array([0.0, 0.0])
        self.rx_position = np.array([100.0, 0.0])
        
        self.distance_m = np.linalg.norm(self.rx_position - self.tx_position)
        
        self.velocity_m_s = 0.1  # Default velocity of nodes in m/s
        self.motion_direction_rad = np.random.uniform(0, 2*np.pi)
        
        self.distance_min_m = 50
        self.distance_max_m = 200
        
        self.channel_state = "LOS"  # Default channel state
        self.blockage_loss_db = 0.0
        
        self.temperature_c = 25.0
        self.humidity_percent = 50.0
        self.pressure_hpa = 1013.25
        
        if self.frequency_ghz == 140:
            self.atmospheric_loss_db_per_km = 0.4
        else:
            self.atmospheric_loss_db_per_km = 2.0
        
        self.is_raining = False
        self.rain_rate_mm_per_hr = 0.0
        self.fog_visibility_m = None
        
        self.interference_dbm = -130.0
        
        self.blockage_timer_s = 0
        self.blockage_duration_s = 0
        
        self.week_duration_days = 7
        self.rain_days_per_week = 3
        self.rain_schedule = set()
        self.current_day = 0
        
        self.nlos_shadowing_db = 0.0
        
        self.rain_windows = {}


        
        assert self.distance_m > 0, "jarak harus positif"
        assert self.beamwidth_deg > 0, "beamwidth harus positif"
        assert 50 <= self.eirp_dbm <= 70, "nilai EIRP diluar rentang realistis THz"

    # =======================================

    def initialize_rain_schedule(self, total_days):
        self.rain_windows = {}

        if self.lab_mode:
            self.rain_schedule = set()
            return

        days = list(range(total_days))
        k = min(self.rain_days_per_week, total_days)

        self.rain_schedule = set(
            np.random.choice(days, size=k, replace=False)
        )

        for d in self.rain_schedule:
            start = np.random.randint(0, 18*60)
            duration = np.random.randint(60, 180)
            self.rain_windows[d] = (start, start + duration)

    # =======================================

    def update_mobility(self, dt):
        """
        Update distance with physically consistent mobility.
        - LAB   : reflective boundary, smooth motion
        - URBAN : reflective + random direction reset
        """

        # small random walk on direction (tracking jitter)
        if self.lab_mode:
            self.motion_direction_rad += np.random.normal(0, 0.01)
        else:
            self.motion_direction_rad += np.random.normal(0, 0.05)

        # radial movement
        delta_d = self.velocity_m_s * dt * np.cos(self.motion_direction_rad)
        delta_d += np.random.normal(0, 0.02)  # micro jitter

        new_d = self.distance_m + delta_d

        # ===============================
        # BOUNDARY HANDLING
        # ===============================
        if new_d > self.distance_max_m:
            # reflect
            excess = new_d - self.distance_max_m
            new_d = self.distance_max_m - excess

            # reverse direction
            self.motion_direction_rad += np.pi

            # urban: randomize heading a bit
            if not self.lab_mode:
                self.motion_direction_rad += np.random.uniform(-np.pi/4, np.pi/4)

        elif new_d < self.distance_min_m:
            excess = self.distance_min_m - new_d
            new_d = self.distance_min_m + excess

            self.motion_direction_rad += np.pi

            if not self.lab_mode:
                self.motion_direction_rad += np.random.uniform(-np.pi/4, np.pi/4)

        self.distance_m = new_d


    # =======================================
    
    def update_environment(self, time_s):
        day = int(time_s // 86400)
        minute_of_day = int((time_s % 86400) // 60)
        self.current_day = day
    
        # rain window logic
        if not self.lab_mode and day in self.rain_windows:
            start, end = self.rain_windows[day]
            if start <= minute_of_day <= end:
                self.is_raining = True
                self.rain_rate_mm_per_hr = np.random.choice([5,10,20,40])
            else:
                self.is_raining = False
                self.rain_rate_mm_per_hr = 0.0
        else:
            self.is_raining = False
            self.rain_rate_mm_per_hr = 0.0


        # =====================
        # TEMPERATURE & HUMIDITY (tetap)
        # =====================
        if self.lab_mode:
            self.temperature_c = 25.0 + np.random.normal(0, 0.3)
            self.humidity_percent = np.clip(50 + np.random.normal(0, 2), 45, 55)
        else:
            hour = (time_s / 3600) % 24
            diurnal = 2.0 * np.sin(2*np.pi * (hour - 6) / 24)
            self.temperature_c = 26.0 + diurnal + np.random.normal(0, 0.8)
            self.humidity_percent = np.clip(80 + np.random.normal(0, 5), 70, 90)

        self.pressure_hpa = 1013.25 + np.random.normal(0, 1.0)

        # # =====================
        # # RAIN (EVENT-BASED)
        # # =====================
        # if not self.lab_mode and day in self.rain_schedule:
        #     self.is_raining = True
        #     self.rain_rate_mm_per_hr = np.random.choice(
        #         [5, 10, 20, 40],
        #         p=[0.3, 0.3, 0.25, 0.15]
        #     )
        # else:
        #     self.is_raining = False
        #     self.rain_rate_mm_per_hr = 0.0

        # =====================
        # FOG (conditional)
        # =====================
        if not self.lab_mode and self.is_raining and self.humidity_percent > 85:
            self.fog_visibility_m = np.random.uniform(50, 300)
        else:
            self.fog_visibility_m = None
            
        if self.channel_state == "NLOS":
            self.nlos_shadowing_db += np.random.normal(0, 0.8)
            self.nlos_shadowing_db = np.clip(self.nlos_shadowing_db, -6, 6)
        else:
            self.nlos_shadowing_db = 0.0
    
    # =======================================
    
    def evaluate_los_nlos(self, dt):
        if self.channel_state == "NLOS":
            self.blockage_timer_s += dt
            
            # 🔥 DYNAMIC BLOCKAGE FLUCTUATION
            self.blockage_loss_db += np.random.normal(0, 1.2)
            self.blockage_loss_db = np.clip(self.blockage_loss_db, 20, 55)

            if self.blockage_timer_s >= self.blockage_duration_s:
                self.channel_state = "LOS"
                self.blockage_loss_db = 0.0
                self.blockage_timer_s = 0
                self.nlos_shadowing_db = 0.0
        else:
            if np.random.rand() < self.p_block_start:
                self.channel_state = "NLOS"
                if self.lab_mode:
                    self.blockage_duration_s = np.random.uniform(1, 5)
                else:
                    self.blockage_duration_s = np.random.uniform(30, 180)

                if self.frequency_ghz == 140:
                    self.blockage_loss_db = np.random.uniform(20, 35)
                else:  # 220 GHz
                    self.blockage_loss_db = np.random.uniform(30, 55)


    # =======================================
    
    def fspl_db(self):
        d = self.distance_m
        f = self.frequency
        assert d > 0, "jarak harus positif untuk perhitungan FSPL"
        
        return 20 * np.log10(d) + 20 * np.log10(f) - 147.55  # FSPL in dB
    
    # =======================================
    
    def atmospheric_loss_db(self):
        d_km = self.distance_m / 1000.0
        
        if self.frequency_ghz == 140:
            gamma_db_per_km = 0.4
        else:
            gamma_db_per_km = 2.0
            
        return gamma_db_per_km * d_km
    
    # =======================================
    
    def rain_attenuation_db(self):
        assert self.frequency_ghz >=100 , "model rain attenuation ini khusus kanal sub-THz"
        
        if not self.is_raining or self.rain_rate_mm_per_hr <=0:
            return 0.0
        
        R = self.rain_rate_mm_per_hr
        d_km = self.distance_m / 1000.0
        
        if self.frequency_ghz == 140:
            k = 2.5
            alpha = 0.65
        else:
            k = 4.0
            alpha = 0.6
            
        
        gamma_r = k * (R ** alpha)
        return gamma_r * d_km
        
    # =======================================
    
    def fog_attenuation_db(self):
        if self.fog_visibility_m is None:
            return 0.0
        
        assert self.frequency_ghz >=100 , "model fog attenuation ini khusus kanal sub-THz"
        
        V_km = self.fog_visibility_m / 1000.0
        d_km = self.distance_m / 1000.0
        
        M = 0.024 * (V_km ** -1.05)
        
        f = self.frequency_ghz
        T = self.temperature_c + 273.15
        
        K_1 = 0.819 * T / (T**2 + (f*0.01)**2)
        
        gamma_fog = K_1 * M
        
        return gamma_fog * d_km
    
    # =======================================
    
    def doppler_hz(self):
        return self.velocity_m_s / self.wavelength_m
    
    # =======================================
        
    def beam_misalignment_loss_db(self):
        # baseline tracking error
        sigma = 0.1
        
        if not self.lab_mode:
            sigma *= 1.8
        
        f_d = self.doppler_hz()
        sigma *= (1 + f_d / 300)

        # worsened tracking in NLOS
        if self.channel_state == "NLOS":
            sigma *= 1.5

        # # worsened tracking in rain
        # if self.is_raining:
        #     sigma *= 1.3

        theta_err = abs(np.random.normal(0, sigma))
        bw = self.beamwidth_deg

        if theta_err <= bw:
            loss_db = 12 * (theta_err / bw) ** 2
        else:
            loss_db = 12 + 15 * np.log10(theta_err / bw)

        return min(loss_db, 20.0)

    # =======================================
        
    def nakagami_fading_linear(self):
        """
        Small-scale fading gain (linear power)
        Rain increases fading severity by reducing m
        """
    
        # ===== BASELINE m =====
        if self.lab_mode and self.channel_state == "LOS":
            m = 8.0                      # lab LOS
        elif self.channel_state == "LOS":
            m = 3.0                      # urban LOS
        else:
            m = 1.2                      # NLOS
    
        # ===== RAIN EFFECT =====
        if self.is_raining:
            if self.channel_state == "LOS":
                m *= 0.6                 # LOS + rain → more fluctuation
            else:
                m *= 0.8                 # NLOS + rain → slightly worse
    
        # safety floor
        m = max(m, 0.7)
    
        h = nakagami.rvs(m)
        return h
    

    
    # =======================================
    
    def total_path_loss_db(self):

        self.last_fspl_db = self.fspl_db()
        self.last_gas_loss_db = self.atmospheric_loss_db()
        self.last_rain_loss_db = self.rain_attenuation_db()
        self.last_fog_loss_db = self.fog_attenuation_db()
        self.last_beam_loss_db = self.beam_misalignment_loss_db()
        self.last_blockage_loss_db = self.blockage_loss_db if self.channel_state == "NLOS" else 0.0

        L_total = (
            self.last_fspl_db
            + self.last_gas_loss_db
            + self.last_rain_loss_db
            + self.last_fog_loss_db
            + self.last_beam_loss_db
            + self.last_blockage_loss_db
            + self.nlos_shadowing_db
        )

        if self.lab_mode:
            margin = 40 if self.frequency_ghz == 140 else 50
        else:
            margin = 80 if self.frequency_ghz == 140 else 100

        if not np.isfinite(L_total):
            self.path_loss_anomaly = True
            self.excess_loss_db = np.inf
        else:
            self.path_loss_anomaly = L_total > self.last_fspl_db + margin
            self.excess_loss_db = max(0.0, L_total - (self.last_fspl_db + margin))

        self.last_path_loss_db = L_total
        return L_total

    
    # =======================================
    
    def noise_power_dbm(self):
        return (-174.0 + 10 * np.log10(self.bandwidth_hz) + self.rx_noise_figure_db)
    
    # =======================================

    def snr_db(self):
        # average received power (dBm)
        L_total = self.total_path_loss_db()
        pr_avg_dbm = self.eirp_dbm + self.rx_antenna_gain_dbi - L_total

        # convert to linear
        pr_avg_mw = 10 ** (pr_avg_dbm / 10)

        # apply Nakagami fading
        h = self.nakagami_fading_linear()
        pr_inst_mw = pr_avg_mw * h

        # back to dBm
        pr_inst_dbm = 10 * np.log10(pr_inst_mw)

        # noise + interference
        n = self.noise_power_dbm()
        I = self.interference_dbm
        Ni = 10 * np.log10(10**(n/10) + 10**(I/10))

        snr = pr_inst_dbm - Ni

        # logging (penting untuk analisis & LSTM)
        self.last_rsrp_dbm = pr_inst_dbm
        self.last_noise_dbm = Ni
        self.last_snr_db = snr
        self.last_fading_gain = h

        return snr

# =======================================

def ensure_output_dir(dir_name="hasil_simulasi"):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

# =======================================
    
def simulate_1day_comparison(environment_mode="lab"):
    dt = 60  # 1 minute
    total_steps = 24 * 60  # 1 day
    
    sim_140 = SimSim(140e9, environment_mode)
    sim_220 = SimSim(220e9, environment_mode)

    # 🔒 pastikan jarak sama (future-proof)
    sim_220.distance_m = sim_140.distance_m

    records = []
    time_s = 0.0
    
    sim_140.initialize_rain_schedule(total_days=1)
    sim_220.rain_schedule = sim_140.rain_schedule


    for step in range(total_steps):
        # ==============================
        # Update environment ONCE
        # ==============================
        sim_140.update_environment(time_s)
        sim_140.evaluate_los_nlos(dt)

        # === COPY BLOCKAGE STATE (WAJIB) ===
        sim_220.channel_state = sim_140.channel_state
        sim_220.blockage_loss_db = sim_140.blockage_loss_db
        sim_220.blockage_timer_s = sim_140.blockage_timer_s
        sim_220.blockage_duration_s = sim_140.blockage_duration_s

        sim_220.rain_windows = sim_140.rain_windows


        # 🔁 copy environment state to 220 GHz
        sim_220.temperature_c = sim_140.temperature_c
        sim_220.pressure_hpa = sim_140.pressure_hpa
        sim_220.humidity_percent = sim_140.humidity_percent
        sim_220.pressure_hpa = sim_140.pressure_hpa
        sim_220.is_raining = sim_140.is_raining
        sim_220.rain_rate_mm_per_hr = sim_140.rain_rate_mm_per_hr
        sim_220.fog_visibility_m = sim_140.fog_visibility_m

        # ==============================
        # LOS / NLOS (independent)
        # ==============================
        # sim_220.evaluate_los_nlos(dt)
        
        sim_140.update_mobility(dt)
        sim_220.update_mobility(dt)

        snr_140 = sim_140.snr_db()
        snr_220 = sim_220.snr_db()

        records.append({
            "time_min": step,
            "distance_m": sim_140.distance_m,

            "snr_140_db": snr_140,
            "snr_220_db": snr_220,
            "snr_gap_db": snr_140 - snr_220,

            "pl_140_db": sim_140.last_path_loss_db,
            "pl_220_db": sim_220.last_path_loss_db,

            "state_140": sim_140.channel_state,
            "state_220": sim_220.channel_state,

            "is_raining": sim_140.is_raining,
            "rain_rate": sim_140.rain_rate_mm_per_hr,
            "fog_visibility_m": sim_140.fog_visibility_m
        })

        time_s += dt

    df = pd.DataFrame(records)
    output_dir = ensure_output_dir("hasil_simulasi")
    filename = f"comparison_140_vs_220_1day_{environment_mode}.csv"
    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")
    return df


# =======================================

def simulate_single_frequency():
    freq_input = input("Pilih frekuensi (140 / 220 GHz): ").strip()
    if freq_input not in ["140", "220"]:
        raise ValueError("Frekuensi harus 140 atau 220 GHz")

    freq_ghz = int(freq_input)
    freq_hz = freq_ghz * 1e9

    days_input = input("Durasi simulasi (1–30 hari): ").strip()
    days = max(1, min(30, int(days_input)))

    environment_mode = input("Environment (lab / urban): ").strip()
    if environment_mode not in ["lab", "urban"]:
        environment_mode = "lab"

    sim = SimSim(freq_hz, environment_mode)
    sim_140 = SimSim(140e9, environment_mode)
    sim_220 = SimSim(220e9, environment_mode)
    
    dt = 60
    total_steps = days * 1440
    time_s = 0.0

    records = []

    sim.initialize_rain_schedule(total_days=days)

    for step in range(total_steps):
        sim.update_environment(time_s)
        sim.evaluate_los_nlos(dt)
        
                # === COPY BLOCKAGE STATE (WAJIB) ===
        sim_220.channel_state = sim_140.channel_state
        sim_220.blockage_loss_db = sim_140.blockage_loss_db
        sim_220.blockage_timer_s = sim_140.blockage_timer_s
        sim_220.blockage_duration_s = sim_140.blockage_duration_s

        sim_220.rain_windows = sim_140.rain_windows


        # 🔁 copy environment state to 220 GHz
        sim_220.temperature_c = sim_140.temperature_c
        sim_220.pressure_hpa = sim_140.pressure_hpa
        sim_220.humidity_percent = sim_140.humidity_percent
        sim_220.pressure_hpa = sim_140.pressure_hpa
        sim_220.is_raining = sim_140.is_raining
        sim_220.rain_rate_mm_per_hr = sim_140.rain_rate_mm_per_hr
        sim_220.fog_visibility_m = sim_140.fog_visibility_m
        
        sim.update_mobility(dt)
        
        snr = sim.snr_db()

        records.append({
            "time_min": step,
            "day": step // 1440,
            "distance_m": sim.distance_m,
            "snr_db": snr,
            "path_loss_db": sim.last_path_loss_db,
            "rsrp_dbm": sim.last_rsrp_dbm,
            "noise_dbm": sim.last_noise_dbm,
            "channel_state": sim.channel_state,
            "blockage_loss_db": sim.blockage_loss_db,
            "beamwidth_deg": sim.beamwidth_deg,
            "temperature_c": sim.temperature_c,
            "humidity_percent": sim.humidity_percent,
            "is_raining": sim.is_raining,
            "rain_rate": sim.rain_rate_mm_per_hr,
            "fog_visibility_m": sim.fog_visibility_m,
            "path_loss_anomaly": sim.path_loss_anomaly,
            "excess_loss_db": sim.excess_loss_db,
            "fading_gain": sim.last_fading_gain
        })

        time_s += dt

    df = pd.DataFrame(records)
    output_dir = ensure_output_dir("hasil_simulasi")
    filename = f"single_{freq_ghz}GHz_{days}days_{environment_mode}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")

    return df

if __name__ == "__main__":
    print("=" * 80)
    print("6G SUB-TERAHERTZ CHANNEL SIMULATOR (ITU-R)")
    print("1. 1-day comparison (140 vs 220 GHz)")
    print("2. Single frequency simulation")
    print("=" * 80)

    choice = input("Choose mode (1/2): ").strip()

    if choice == "1":
        env = input("Environment (lab / urban): ").strip()
        if env not in ["lab", "urban"]:
            env = "lab"
        simulate_1day_comparison(env)

    elif choice == "2":
        simulate_single_frequency()

    else:
        print("Invalid choice. Running default comparison.")
        simulate_1day_comparison("lab")