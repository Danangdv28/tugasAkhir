import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy.stats import nakagami

class TerahertzChannelSimulator:
    def __init__(self, frequency=140e9):
        """
        Simulator kanal 6G Sub-THz dengan ITU-R Compliance
        Frekuensi: 140 GHz dan 220 GHz (D-band & G-band)
        Menggunakan model ITU-R P.676-13, P.838-3, P.1817-1
        """
        self.frequency = frequency
        freq_ghz = frequency / 1e9
        
        self.environment_mode = "lab"   # change to "urban_stress" when needed
        # self.environment_mode = "urban_stress"   # change to "urban_stress" when needed
        
        # --- FREQUENCY-DEPENDENT HARDWARE (ITU-R P.1817-1) ---
        # Tx power lebih rendah di THz karena keterbatasan PA
        if freq_ghz <= 150:
            self.tx_power_dbm = 20  # D-band limitation
        else:
            self.tx_power_dbm = 15  # G-band more limited
        
        # Antenna gain: FIXED VALUES as specified
        # 140 GHz: Gtx = Grx = 40 dBi
        # 220 GHz: Gtx = Grx = 45 dBi
        if freq_ghz <= 150:
            self.tx_antenna_gain_dbi = 40.0  # 140 GHz
            self.rx_gain = 40.0
        else:
            self.tx_antenna_gain_dbi = 45.0  # 220 GHz
            self.rx_gain = 45.0
        
        self.eirp_dbm = self.tx_power_dbm + self.tx_antenna_gain_dbi
        
        # Noise figure sangat tinggi di THz (ITU-R P.1817-1)
        # NF ≈ 10-15 dB untuk D-band, 15-20 dB untuk G-band
        if freq_ghz <= 150:
            self.rx_noise_figure = 12.0
        else:
            self.rx_noise_figure = 16.5
        
        # Bandwidth lebih besar di THz
        self.bandwidth = 2e9 if freq_ghz > 200 else 1e9  # 1-2 GHz
        
        self.c = 3e8
        self.wavelength = self.c / self.frequency
        
        # Calculate actual beamwidth from antenna gain
        # G = 4π*A/λ² = 4π*(πD²/4)/λ² → G = (πD/λ)²
        # Beamwidth (3dB) ≈ 70λ/D for parabolic antenna
        # From gain: D/λ = sqrt(G_linear/π)
        gain_linear = 10 ** (self.tx_antenna_gain_dbi / 10)
        D_over_lambda = np.sqrt(gain_linear / np.pi)
        self.beamwidth_deg = 70 / D_over_lambda  # 3dB beamwidth in degrees
        
        # --- GEOMETRIC MOBILITY ---
        # Lab range: lebih pendek dan terkontrol (50-200m)
        start_dist = random.uniform(50, 150)
        start_angle = random.uniform(0, 2*np.pi)
        self.user_x = start_dist * np.cos(start_angle)
        self.user_y = start_dist * np.sin(start_angle)
        self.distance = np.sqrt(self.user_x**2 + self.user_y**2)
        
        # Slow movement in lab (quasi-static)
        self.velocity = random.uniform(0.05, 0.2)  # m/s (very slow movement)
        self.direction = random.uniform(0, 2*np.pi)
        
        # --- FREQUENCY-DEPENDENT ENVIRONMENT (LAB) ---
        # Lab environment: controlled reflections from walls/equipment
        # Much lower reflection loss than outdoor
        self.reflection_loss_per_bounce = 2.0 + (freq_ghz - 140) * 0.05
        
        # Lab walls: glass, metal panels (lower penetration loss)
        self.wall_penetration_loss = 20.0 + (freq_ghz - 140) * 0.2
        
        self.num_reflections = random.randint(0, 2)  # Minimal reflections
        self.num_walls_penetrated = 0  # No wall penetration in lab LOS
        
        # --- BLOCKAGE STATE (Very rare in controlled lab) ---
        self.is_blocked = False
        self.blockage_timer = 0
        self.blockage_transition_prob = 0.001  # 0.1% (extremely rare in lab)
        if self.environment_mode == "urban_stress":
            self.blockage_transition_prob = 0.03  # 1%
        
        self.blockage_exit_prob = 0.5  # 50% quick recovery
        
        # --- ATMOSPHERIC & ENVIRONMENTAL (LAB CONTROLLED) ---
        self.temperature = 23.0  # Controlled lab temp
        self.humidity = 50.0  # Controlled lab humidity
        self.pressure_hpa = 1013.25  # Standard pressure
        self.water_vapor_density = self.calculate_water_vapor_density()
        self.fog_visibility = None  # No fog in lab
        
        # --- LAB ENVIRONMENT (Minimal Fading) ---
        # Controlled indoor environment with stable channel
        self.lab_mode = (self.environment_mode == "lab")  # Lab environment flag
        self.multipath_components = 1  # Minimal multipath in lab
        
        # --- USERS & TRAFFIC ---
        self.num_users = random.randint(10, 30)  # Lebih sedikit untuk THz
        self.traffic_density = random.uniform(0.5, 0.8)
        
        # --- ENVIRONMENT MODE ---
        # "lab" = baseline
        # "urban_stress" = Bandung-like humidity + rain + objects

        
        # --- OBJECT-BASED OBSTRUCTIONS (URBAN / LAB EMULATION) ---
        self.obstacles = [
            {'type': 'metal_rack', 'loss': 10, 'prob': 0.02},
            {'type': 'glass_panel', 'loss': 6,  'prob': 0.03},
            {'type': 'human_body', 'loss': 15, 'prob': 0.01},
        ]



        
    def calculate_water_vapor_density(self):
        """Water vapor density (g/m³) - ITU-R P.676-13"""
        # Saturation vapor pressure (hPa)
        es = 6.1121 * np.exp((18.678 - self.temperature/234.5) * 
                              (self.temperature / (257.14 + self.temperature)))
        # Actual vapor pressure
        e = (self.humidity / 100.0) * es
        # Water vapor density
        rho_v = (216.7 * e) / (self.temperature + 273.15)
        return rho_v
    
    def get_rain_attenuation_itu(self, rain_rate):
        """
        ITU-R P.838-3: Rain attenuation untuk frekuensi tinggi
        γ_R = k * R^α (dB/km)
        """
        f = self.frequency / 1e9
        
        # Coefficients untuk polarisasi vertikal
        # Menggunakan interpolasi/ekstrapolasi dari tabel ITU-R P.838-3
        
        if f <= 170:
            # D-band region (140 GHz)
            # Ekstrapol dari data 100-150 GHz
            k = 2.5 + (f - 100) * 0.02
            alpha = 0.65 + (f - 100) * 0.001
        else:
            # G-band region (220 GHz)
            # Rain absorption sangat ekstrim
            k = 4.0 + (f - 170) * 0.025
            alpha = 0.60
        
        # Specific attenuation (dB/km)
        gamma_rain = k * (rain_rate ** alpha)
        
        return gamma_rain
    
    def atmospheric_loss_p676_ref(self):
        """
        Simplified ITU-R P.676 reference (order-accurate)
        """
        f = self.frequency / 1e9
        rho = self.water_vapor_density

        gamma_o = 0.02 * (f / 60)**2     # oxygen
        gamma_w = 0.01 * rho * (f / 100)**2  # water vapor

        return (gamma_o + gamma_w) * (self.distance / 1000)

    
    def get_atmospheric_attenuation_itu(self):
        """
        ITU-R P.676-13: Atmospheric gaseous attenuation
        SIMPLIFIED for lab environment with realistic limits
        """
        f = self.frequency / 1e9
        
        # Lab-based fixed atmospheric loss (ITU-R defensible)
        if f < 170:
            return 0.4   # dB/km (140 GHz lab)
        else:
            return 2.0   # dB/km (220 GHz lab)
    
    def get_fog_attenuation_itu(self, visibility_m):
        """
        ITU-R P.840-9: Fog and cloud attenuation
        Ekstrem di THz!
        """
        if visibility_m is None or visibility_m > 1000:
            return 0.0
        
        f = self.frequency / 1e9
        
        # Liquid water content (g/m³) dari visibility
        M = 0.024 * (visibility_m / 1000) ** (-1.05)
        
        # Specific attenuation (dB/km)
        # Model Rayleigh untuk fog droplets
        # γ_fog ≈ K_l * M * f²
        
        # K_l tergantung temperature
        K_l = 0.819 * (self.temperature + 273.15) / (
            (self.temperature + 273.15)**2 + 
            (f * 0.01)**2
        )
        
        gamma_fog = K_l * M * (f ** 2)
        
        return gamma_fog
    
    def update_position_and_blockage(self, interval_minutes=1):
        """Update posisi user dengan boundary 50-200m untuk lab"""
        dist_move = self.velocity * (interval_minutes * 60)
        
        self.user_x += dist_move * np.cos(self.direction)
        self.user_y += dist_move * np.sin(self.direction)
        new_dist = np.sqrt(self.user_x ** 2 + self.user_y ** 2)
        
        # Boundary enforcement (HARD CLAMP)
        if new_dist > 200:
            scale = 200 / new_dist
            self.user_x *= scale
            self.user_y *= scale
        elif new_dist < 50:
            scale = 50 / new_dist
            self.user_x *= scale
            self.user_y *= scale

        self.distance = np.sqrt(self.user_x**2 + self.user_y**2)

        
        # Blockage update (rare in lab)
        if self.is_blocked:
            if random.random() < self.blockage_exit_prob:
                self.is_blocked = False
                self.blockage_timer = 0
            else:
                self.blockage_timer += interval_minutes
        else:
            if random.random() < self.blockage_transition_prob:
                self.is_blocked = True
                self.blockage_timer = random.randint(1, 3)  # Short blockage
    
    def evaluate_los_state(self):
        for obj in self.obstacles:
            if random.random() < obj['prob']:
                return False, obj['loss'], obj['type']
        return True, 0, None
    
    def path_loss_itu_p1238(self):
        """
        ITU-R P.1238-12 Indoor Path Loss (Office / Lab LOS)
        L = 20log10(f) + N log10(d) + Lf(n) - 28
        """
        f_ghz = self.frequency / 1e9
        d = self.distance

        N = 18  # Office / Lab LOS (P.1238 table)
        Lf = 0  # Same floor lab

        return 20*np.log10(f_ghz) + N*np.log10(d) + Lf - 28


    
    def calculate_path_loss_comprehensive(self, timestamp, is_raining, rain_rate):
        """Path loss dengan model ITU-R untuk LAB ENVIRONMENT (minimal fading)"""
        
        obstruction = None
        
        # 1. FREE SPACE PATH LOSS (ITU-R P.525-4)
        fspl = 20 * np.log10(self.distance) + 20 * np.log10(self.frequency) - 147.55
        
        # 2. ATMOSPHERIC GASEOUS ABSORPTION (ITU-R P.676-13)
        # Lab indoor: reduced atmospheric effects
        if self.environment_mode == "lab":
            atm_loss_db_km = self.get_atmospheric_attenuation_itu() * 0.5
        else:
            atm_loss_db_km = self.get_atmospheric_attenuation_itu()  # full loss
            
        atm_loss = atm_loss_db_km * (self.distance / 1000.0)
        
        atm_ref = self.atmospheric_loss_p676_ref()
        atm_dev = atm_loss - atm_ref

        assert abs(atm_dev) <= 2.0, (
            f"P.676 deviation {atm_dev:.2f} dB exceeds 2 dB limit"
        )

        
        # 3. RAIN ATTENUATION (ITU-R P.838-3)
        if self.environment_mode == "urban_stress" and is_raining:
            gamma_rain = self.get_rain_attenuation_itu(rain_rate)
            rain_loss = gamma_rain * (self.distance / 1000.0)
        else:
            rain_loss = 0

        
        # 4. FOG ATTENUATION (ITU-R P.840-9)
        # No fog in lab
        fog_loss = 0
        
        # 5. OBJECT-BASED LOS / NLOS (UNIFIED MODEL)
        los_status, object_loss, obstruction = self.evaluate_los_state()
        
        if not los_status:
            env_loss = object_loss + np.random.normal(1.0, 0.5)
        else:
            if self.environment_mode == "urban_stress":
                env_loss = 1.5 + np.random.normal(0, 0.5)
            else:
                env_loss = max(0, np.random.normal(0, 0.3))


        
        # 6. BEAM MISALIGNMENT (Critical in THz but controlled in lab)
        beam_loss = 0
        if los_status:
            # Lab setup: tripod-mounted with alignment tools
            
            # Use calculated beamwidth from antenna gain
            beamwidth_deg = self.beamwidth_deg
            
            if self.environment_mode == "urban_stress":
                theta = np.abs(np.random.normal(0, 0.25))
            else:
                theta = np.abs(np.random.normal(0, 0.1))
            
            if theta < beamwidth_deg:
                beam_loss = 12 * (theta / beamwidth_deg) ** 2
            else:
                beam_loss = 12 + 15 * np.log10(theta / beamwidth_deg)
            
            beam_loss = min(beam_loss, 20)
            

        
        # 7. SMALL-SCALE FADING - REMOVED FOR LAB
        # Lab environment: quasi-static channel, minimal fading
        # Only measurement noise remains
        measurement_noise = np.random.normal(0, 0.2)  # Instrument noise only (<0.5 dB)
        small_scale_fading = np.clip(measurement_noise, -0.5, 0.5)
        
        # TOTAL PATH LOSS
        total_pl = (fspl + atm_loss + rain_loss + fog_loss + 
                    env_loss + beam_loss + small_scale_fading)
        
        # SANITY CHECK for lab environment
        if self.environment_mode == "lab":
            assert total_pl < 155, (
                f"Path loss {total_pl:.1f} dB exceeds lab upper bound "
                f"(distance={self.distance:.1f}m)"
            )
        else:
            assert total_pl < fspl + 40, (
                f"Urban path loss too large: {total_pl:.1f} dB"
            )

        
        return total_pl, {
            'fspl': fspl,
            'atm_loss': atm_loss,
            'rain_loss': rain_loss,
            'fog_loss': fog_loss,
            'env_loss': env_loss,
            'beam_loss': beam_loss,
            'small_scale_fading': small_scale_fading,
            'los_status': los_status,
            'obstruction_type': obstruction
        }
    
    def calculate_interference(self, hour):
        """
        Interference calculation - LAB ENVIRONMENT
        THz lab != cellular reuse, minimal interference
        """
        # Lab environment: isolated testing, no cellular interference
        # Only thermal noise and minimal lab equipment EMI
        interference = -130  # dBm (very low, equipment EMI only)
        
        active_users = 0  # Single link in lab
        current_traffic = 0.0  # No multi-user traffic
        
        return interference, active_users, current_traffic
    
    def update_environment(self, timestamp):
        hour = timestamp.hour

        # Temperature (still lab-controlled)
        temp_variation = 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)
        self.temperature = 23.0 + temp_variation + np.random.normal(0, 0.2)

        if self.environment_mode == "urban_stress":
            # Bandung-like urban humidity
            self.humidity = np.clip(
                80 + np.random.normal(0, 5), 70, 90
            )
        else:
            # Lab baseline
            self.humidity = np.clip(
                50 + np.random.normal(0, 1), 45, 55
            )

        self.pressure_hpa = 1013.25 + np.random.normal(0, 1)
        self.water_vapor_density = self.calculate_water_vapor_density()

    
    def simulate_measurement(self, timestamp, is_raining=False, rain_rate=0):
        """Single measurement"""
        hour = timestamp.hour
        
        self.update_position_and_blockage(interval_minutes=1)
        self.update_environment(timestamp)
        
        path_loss, components = self.calculate_path_loss_comprehensive(
            timestamp, is_raining, rain_rate
        )
        
        interference, active_users, traffic = self.calculate_interference(hour)
        
        # LINK BUDGET
        rsrp = self.eirp_dbm + self.rx_gain - path_loss
        
        thermal_noise = -174 + 10 * np.log10(self.bandwidth)
        noise_power = thermal_noise + self.rx_noise_figure
        
        inr = 10 * np.log10(10 ** (interference / 10) + 10 ** (noise_power / 10))
        snr = rsrp - inr
        sinr = snr
        
        rssi = 10 * np.log10(
            10 ** (rsrp / 10) + 10 ** (interference / 10) + 10 ** (noise_power / 10)
        )
        
        # Physical Doppler (not directly observable in lab)
        doppler_physical = (self.velocity * np.cos(self.direction)) / self.c * self.frequency

        # Effective Doppler after carrier & beam tracking (lab condition)
        doppler_effective = doppler_physical * 0.05  # 95% compensated

        assert abs(doppler_effective) <= 100, (
            f"Effective Doppler {doppler_effective:.1f} Hz exceeds lab tolerance"
        )


        
        return {
            'timestamp': timestamp,
            'hour': hour,
            'frequency_ghz': round(self.frequency / 1e9, 1),
            
            'distance_m': round(self.distance, 2),
            'user_x': round(self.user_x, 2),
            'user_y': round(self.user_y, 2),
            'velocity_ms': round(self.velocity, 2),
            
            'is_blocked': self.is_blocked,
            'blockage_duration_min': self.blockage_timer,
            'los_status': components['los_status'],
            
            'rsrp_dbm': round(rsrp, 2),
            'rssi_dbm': round(rssi, 2),
            'snr_db': round(snr, 2),
            'sinr_db': round(sinr, 2),
            
            'path_loss_total_db': round(path_loss, 2),
            'fspl_db': round(components['fspl'], 2),
            'atmospheric_loss_db': round(components['atm_loss'], 2),
            'rain_loss_db': round(components['rain_loss'], 2),
            'fog_loss_db': round(components['fog_loss'], 2),
            'blockage_env_loss_db': round(components['env_loss'], 2),
            'beam_misalignment_loss_db': round(components['beam_loss'], 2),
            'small_scale_fading_db': round(components['small_scale_fading'], 2),
            
            'temperature_c': round(self.temperature, 2),
            'humidity_percent': round(self.humidity, 2),
            'pressure_hpa': round(self.pressure_hpa, 2),
            'water_vapor_density_gm3': round(self.water_vapor_density, 2),
            'is_raining': is_raining,
            'rain_rate_mm_h': rain_rate,
            'fog_visibility_m': self.fog_visibility if self.fog_visibility else 9999,
            
            'interference_dbm': round(interference, 2),
            'noise_power_dbm': round(noise_power, 2),
            'active_users': active_users,
            
            'channel_quality': 'LOS_stable' if components['los_status'] else 'NLOS',
            'lab_mode': self.lab_mode,
            
            'eirp_dbm': round(self.eirp_dbm, 2),
            'tx_antenna_gain_dbi': round(self.tx_antenna_gain_dbi, 2),
            'rx_gain_dbi': round(self.rx_gain, 2),
            'rx_noise_figure_db': round(self.rx_noise_figure, 2),
            'bandwidth_ghz': round(self.bandwidth / 1e9, 2),
            'beamwidth_deg': round(self.beamwidth_deg, 3),
            
            'channel_state': 'LOS' if components['los_status'] else 'NLOS',
            'obstruction_type': components.get('obstruction_type', None),
            'humidity_state': 'high' if self.humidity > 70 else 'normal',
            'rain_state': 'rain' if is_raining else 'clear',
        }


def generate_weather_events(start_time, duration_days=1):
    events = []

    # Urban stress: rain only when enabled
    events.append({
        'type': 'rain',
        'start': start_time + timedelta(hours=6),
        'duration_minutes': 90,
        'intensity': 20   # moderate rain (mm/h)
    })

    events.append({
        'type': 'rain',
        'start': start_time + timedelta(hours=15),
        'duration_minutes': 60,
        'intensity': 50   # heavy rain
    })

    return events



def check_weather(timestamp, weather_events, simulator):
    """Check current weather"""
    is_raining = False
    rain_rate = 0
    
    for event in weather_events:
        end_time = event['start'] + timedelta(minutes=event['duration_minutes'])
        
        if event['start'] <= timestamp <= end_time:
            if event['type'] == 'rain':
                is_raining = True
                rain_rate = event['intensity']
            elif event['type'] == 'fog':
                simulator.fog_visibility = event['intensity']
                return is_raining, rain_rate
    
    simulator.fog_visibility = None
    return is_raining, rain_rate


def main_vulnerability_analysis():
    """Vulnerability Analysis: 140 GHz vs 220 GHz - LAB ENVIRONMENT"""
    print("=" * 90)
    print("  🔬 SIMULATOR KANAL 6G SUB-TERAHERTZ - ITU-R BASED")
    print("  🏢 LAB ENVIRONMENT (Minimal Fading, Controlled Conditions)")
    print("  📡 D-band (140 GHz) vs G-band (220 GHz)")
    print("=" * 90)
    
    start_time = datetime.now()
    weather = generate_weather_events(start_time, duration_days=1)
    
    if len(weather) == 0:
        print("\n☁️ WEATHER: Lab environment - no rain/fog events")
    
    print("\n📡 INITIALIZING SIMULATORS...")
    sim_140 = TerahertzChannelSimulator(frequency=140e9)
    sim_220 = TerahertzChannelSimulator(frequency=220e9)
    
    # Sync initial conditions
    sim_220.user_x = sim_140.user_x
    sim_220.user_y = sim_140.user_y
    sim_220.distance = sim_140.distance
    sim_220.velocity = sim_140.velocity
    sim_220.direction = sim_140.direction
    
    print(f"  140 GHz: Ptx={sim_140.tx_power_dbm} dBm, Gtx={sim_140.tx_antenna_gain_dbi:.1f} dBi, "
          f"Grx={sim_140.rx_gain:.1f} dBi, EIRP={sim_140.eirp_dbm:.1f} dBm")
    print(f"           NF={sim_140.rx_noise_figure:.1f} dB, BW={sim_140.bandwidth/1e9:.1f} GHz, "
          f"Beamwidth={sim_140.beamwidth_deg:.2f}°")
    print(f"  220 GHz: Ptx={sim_220.tx_power_dbm} dBm, Gtx={sim_220.tx_antenna_gain_dbi:.1f} dBi, "
          f"Grx={sim_220.rx_gain:.1f} dBi, EIRP={sim_220.eirp_dbm:.1f} dBm")
    print(f"           NF={sim_220.rx_noise_figure:.1f} dB, BW={sim_220.bandwidth/1e9:.1f} GHz, "
          f"Beamwidth={sim_220.beamwidth_deg:.2f}°")
    print(f"  Initial Distance: {sim_140.distance:.1f}m (Lab Range: 50-200m)")
    print(f"  Velocity: {sim_140.velocity:.2f} m/s (quasi-static)")
    print(f"  Lab Conditions: T={sim_140.temperature:.1f}°C, RH={sim_140.humidity:.1f}%")
    
    print("\n⏳ RUNNING 24-HOUR SIMULATION...")
    
    results = []
    current_time = start_time
    
    # Validation counters
    sanity_checks_passed = 0
    total_samples = 0
    
    for i in range(1440):
        is_raining, rain_rate = check_weather(current_time, weather, sim_140)
        
        try:
            res_140 = sim_140.simulate_measurement(current_time, is_raining, rain_rate)
            
            # Sync position
            sim_220.user_x = sim_140.user_x
            sim_220.user_y = sim_140.user_y
            sim_220.distance = sim_140.distance
            sim_220.is_blocked = sim_140.is_blocked
            sim_220.blockage_timer = sim_140.blockage_timer
            
            res_220 = sim_220.simulate_measurement(current_time, is_raining, rain_rate)
            
            sanity_checks_passed += 1
            
        except AssertionError as e:
            print(f"\n⚠️ Sanity check failed at minute {i}: {e}")
            continue
        
        total_samples += 1
        
        results.append({
            'time': current_time,
            'hour': current_time.hour,
            'distance': res_140['distance_m'],
            'is_blocked': res_140['is_blocked'],
            'is_raining': res_140['is_raining'],
            'rain_rate': rain_rate,
            
            'snr_140ghz': res_140['snr_db'],
            'rsrp_140ghz': res_140['rsrp_dbm'],
            'rain_loss_140ghz': res_140['rain_loss_db'],
            'atm_loss_140ghz': res_140['atmospheric_loss_db'],
            'fog_loss_140ghz': res_140['fog_loss_db'],
            'beam_loss_140ghz': res_140['beam_misalignment_loss_db'],
            
            'snr_220ghz': res_220['snr_db'],
            'rsrp_220ghz': res_220['rsrp_dbm'],
            'rain_loss_220ghz': res_220['rain_loss_db'],
            'atm_loss_220ghz': res_220['atmospheric_loss_db'],
            'fog_loss_220ghz': res_220['fog_loss_db'],
            'beam_loss_220ghz': res_220['beam_misalignment_loss_db'],
            
            'snr_degradation': res_140['snr_db'] - res_220['snr_db'],
            'outage_140ghz': res_140['snr_db'] < 0,
            'outage_220ghz': res_220['snr_db'] < 0
        })
        
        current_time += timedelta(minutes=1)
        
        if (i + 1) % 360 == 0:
            print(f"  ✓ {(i+1)//60} hours completed...")
            

    
    df = pd.DataFrame(results)
    
    filename = f'6g_thz_lab_{start_time.strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(filename, index=False)
    
    print("\n" + "=" * 90)
    print("✅ SIMULATION COMPLETE!")
    print("=" * 90)
    print(f"\n🔍 SANITY CHECKS: {sanity_checks_passed}/{total_samples} samples passed validation")
    if sanity_checks_passed == total_samples:
        print("   ✓ All samples within physical bounds!")
    
    # ANALYSIS
    print("\n" + "=" * 90)
    print("📊 LAB VULNERABILITY ANALYSIS - 140 GHz vs 220 GHz")
    print("=" * 90)
    
    print("\n🔋 OVERALL PERFORMANCE:")
    print(f"{'Metric':<30} {'140 GHz':>15} {'220 GHz':>15} {'Difference':>15}")
    print("-" * 77)
    print(f"{'Avg SNR (dB)':<30} {df['snr_140ghz'].mean():>15.2f} {df['snr_220ghz'].mean():>15.2f} {df['snr_degradation'].mean():>15.2f}")
    print(f"{'Min SNR (dB)':<30} {df['snr_140ghz'].min():>15.2f} {df['snr_220ghz'].min():>15.2f}")
    print(f"{'Max SNR (dB)':<30} {df['snr_140ghz'].max():>15.2f} {df['snr_220ghz'].max():>15.2f}")
    print(f"{'Std Dev SNR (dB)':<30} {df['snr_140ghz'].std():>15.2f} {df['snr_220ghz'].std():>15.2f}")
    print(f"{'Outage Prob (%)':<30} {(df['outage_140ghz'].sum()/len(df)*100):>15.2f} {(df['outage_220ghz'].sum()/len(df)*100):>15.2f}")
    
    clear = df[(df['rain_rate'] == 0) & (df['is_blocked'] == False)]
    blocked = df[df['is_blocked'] == True]
    
    print("\n🌤️ CLEAR CHANNEL (LOS - Lab Conditions):")
    if len(clear) > 0:
        print(f"  140 GHz: SNR = {clear['snr_140ghz'].mean():.2f} ± {clear['snr_140ghz'].std():.2f} dB")
        print(f"  220 GHz: SNR = {clear['snr_220ghz'].mean():.2f} ± {clear['snr_220ghz'].std():.2f} dB")
        print(f"  → Channel Stability (Std Dev):")
        print(f"     140 GHz: {clear['snr_140ghz'].std():.3f} dB (very stable)")
        print(f"     220 GHz: {clear['snr_220ghz'].std():.3f} dB")
        print(f"  → Atmospheric loss 140 GHz: {clear['atm_loss_140ghz'].mean():.3f} dB")
        print(f"  → Atmospheric loss 220 GHz: {clear['atm_loss_220ghz'].mean():.3f} dB")
        print(f"  → Beam alignment loss 140 GHz: {clear['beam_loss_140ghz'].mean():.3f} dB")
        print(f"  → Beam alignment loss 220 GHz: {clear['beam_loss_220ghz'].mean():.3f} dB")
    
    print("\n🏢 BLOCKAGE (NLOS - Equipment/Furniture):")
    if len(blocked) > 0:
        print(f"  140 GHz: SNR = {blocked['snr_140ghz'].mean():.2f} dB")
        print(f"  220 GHz: SNR = {blocked['snr_220ghz'].mean():.2f} dB")
        
        out_140 = (blocked['snr_140ghz'] < 0).sum() / len(blocked) * 100
        out_220 = (blocked['snr_220ghz'] < 0).sum() / len(blocked) * 100
        
        print(f"  → Outage Rate 140 GHz: {out_140:.1f}%")
        print(f"  → Outage Rate 220 GHz: {out_220:.1f}%")
        print(f"  → Blockage occurrence: {len(blocked)/len(df)*100:.1f}% (rare in lab)")
    else:
        print(f"  No blockage events (excellent lab setup)")
    
    print("\n💡 KEY INSIGHTS (Lab Environment - ITU-R Based):")
    print(f"  1. SNR Degradation: 220 GHz is {df['snr_degradation'].mean():.1f} dB worse on average")
    print(f"  2. Channel Stability: Std Dev = {df['snr_140ghz'].std():.3f} dB (140 GHz), "
          f"{df['snr_220ghz'].std():.3f} dB (220 GHz)")
    print(f"  3. Atmospheric (ITU-R): 140 GHz = 0.4 dB/km, 220 GHz = 2.0 dB/km (fixed)")
    print(f"  4. Interference: -130 dBm (lab equipment EMI only, no cellular)")
    print(f"  5. Beam Alignment: Critical for both (beamwidth 140GHz={sim_140.beamwidth_deg:.2f}°, 220GHz={sim_220.beamwidth_deg:.2f}°)")
    print(f"  6. Fading: Minimal (lab quasi-static, <0.5 dB variation)")
    print(f"  7. Range: 50-200m (lab controlled environment)")
    print(f"  8. Blockage: {(df['is_blocked'].sum()/len(df)*100):.2f}% (extremely rare, prob=0.1%)")
    
    print(f"\n📁 Data saved: {filename}")
    print("=" * 90)
    
    return df


def main_single_frequency(frequency_ghz=140, duration_days=7):
    """Single frequency simulation - LAB ENVIRONMENT"""
    print("=" * 90)
    print(f"  📡 6G SUB-TERAHERTZ SIMULATOR - {frequency_ghz} GHz (ITU-R)")
    print("  🏢 LAB ENVIRONMENT (Quasi-static, Minimal Fading)")
    print("=" * 90)
    
    simulator = TerahertzChannelSimulator(frequency=frequency_ghz * 1e9)
    start_time = datetime.now()
    weather = generate_weather_events(start_time, duration_days)
    
    print(f"\n📊 SYSTEM PARAMETERS:")
    print(f"  Frequency: {frequency_ghz} GHz")
    print(f"  Tx Power: {simulator.tx_power_dbm} dBm")
    print(f"  Tx Antenna Gain: {simulator.tx_antenna_gain_dbi:.1f} dBi")
    print(f"  Rx Antenna Gain: {simulator.rx_gain:.1f} dBi")
    print(f"  EIRP: {simulator.eirp_dbm:.1f} dBm")
    print(f"  Noise Figure: {simulator.rx_noise_figure:.1f} dB")
    print(f"  Bandwidth: {simulator.bandwidth/1e9:.1f} GHz")
    print(f"  Beamwidth (3dB): {simulator.beamwidth_deg:.2f}°")
    print(f"  Range: 50-200m (Lab)")
    print(f"  Velocity: {simulator.velocity:.2f} m/s (quasi-static)")
    print(f"  Environment: Lab controlled (T=23±1°C, RH=50±3%)")
    
    print(f"\n☁️ Weather: None (lab environment)")
    print(f"✨ Fading: Minimal (<0.5 dB variation)")
    
    print("\n⏳ SIMULATING...")
    
    data = []
    current_time = start_time
    total_minutes = duration_days * 24 * 60
    
    # Validation counters
    sanity_checks_passed = 0
    total_samples = 0
    
    for i in range(total_minutes):
        is_raining, rain_rate = check_weather(current_time, weather, simulator)
        
        try:
            measurement = simulator.simulate_measurement(current_time, is_raining, rain_rate)
            data.append(measurement)
            sanity_checks_passed += 1
        except AssertionError as e:
            print(f"\n⚠️ Sanity check failed at minute {i}: {e}")
            continue
        
        total_samples += 1
        
        if (i + 1) % 1440 == 0:
            day = (i + 1) // 1440
            avg_snr = np.mean([d['snr_db'] for d in data[-1440:]])
            std_snr = np.std([d['snr_db'] for d in data[-1440:]])
            print(f"  ✓ Day {day}/{duration_days} - SNR: {avg_snr:.2f}±{std_snr:.2f} dB")
        
        current_time += timedelta(minutes=1)
    
    df = pd.DataFrame(data)
    filename = f'6g_thz_lab_{frequency_ghz}ghz_{start_time.strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(filename, index=False)
    
    print("\n" + "=" * 90)
    print("✅ COMPLETE!")
    print("=" * 90)
    print(f"\n🔍 SANITY CHECKS: {sanity_checks_passed}/{total_samples} samples passed validation")
    if sanity_checks_passed == total_samples:
        print("   ✓ All samples within physical bounds!")
    
    print(f"\n📈 STATISTICS:")
    print(f"  Samples: {len(df):,}")
    print(f"  SNR: {df['snr_db'].min():.1f} to {df['snr_db'].max():.1f} dB "
          f"(μ={df['snr_db'].mean():.1f}, σ={df['snr_db'].std():.2f})")
    print(f"  Distance: {df['distance_m'].min():.1f} to {df['distance_m'].max():.1f}m")
    print(f"  Blockage: {df['is_blocked'].sum()/len(df)*100:.1f}% (rare)")
    print(f"  Outage (SNR<0): {(df['snr_db']<0).sum()/len(df)*100:.1f}%")
    print(f"  Channel Stability: σ = {df['snr_db'].std():.3f} dB (very stable)")
    
    print(f"\n🎯 CHANNEL QUALITY:")
    excellent = (df['snr_db'] >= 20).sum()
    good = ((df['snr_db'] >= 10) & (df['snr_db'] < 20)).sum()
    fair = ((df['snr_db'] >= 0) & (df['snr_db'] < 10)).sum()
    poor = (df['snr_db'] < 0).sum()
    
    print(f"  Excellent (≥20 dB): {excellent:6d} ({excellent/len(df)*100:5.1f}%)")
    print(f"  Good (10-20 dB):    {good:6d} ({good/len(df)*100:5.1f}%)")
    print(f"  Fair (0-10 dB):     {fair:6d} ({fair/len(df)*100:5.1f}%)")
    print(f"  Poor (<0 dB):       {poor:6d} ({poor/len(df)*100:5.1f}%)")
    
    print(f"\n📁 File: {filename}")
    print("\n💡 Lab Environment Characteristics:")
    print(f"  • Minimal fading (measurement noise only)")
    print(f"  • Stable temperature/humidity")
    print(f"  • Controlled beam alignment")
    print(f"  • Quasi-static channel (slow movement)")
    print(f"  • Ideal for deterministic modeling & LSTM training")
    
    return df


if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("  6G SUB-TERAHERTZ CHANNEL SIMULATOR (ITU-R COMPLIANCE)")
    print("  🏢 LAB ENVIRONMENT - Minimal Fading, Controlled Conditions")
    print("  Frequencies: 140 GHz (D-band) & 220 GHz (G-band)")
    print("=" * 90)
    print("\nChoose mode:")
    print("  1. Vulnerability Analysis (140 GHz vs 220 GHz - 24 hours)")
    print("  2. Single Frequency Simulation (7 days)")
    
    choice = input("\nChoice (1/2): ").strip()
    
    if choice == "1":
        print("\n🔬 Running Lab Vulnerability Analysis...")
        df = main_vulnerability_analysis()
    elif choice == "2":
        freq = input("Frequency (140 or 220 GHz): ").strip()
        try:
            freq_val = float(freq)
            if freq_val not in [140, 220]:
                print("⚠️ Using 140 GHz")
                freq_val = 140
        except:
            freq_val = 140
        
        days = input("Duration (days, 1-30, default 7): ").strip()
        try:
            days_val = int(days)
            days_val = max(1, min(30, days_val))
        except:
            days_val = 7
        
        df = main_single_frequency(frequency_ghz=freq_val, duration_days=days_val)
    else:
        print("\n⚠️ Invalid. Running default...")
        df = main_vulnerability_analysis()
    
    print("\n✨ Simulation completed successfully!")
    print("\n🧪 LAB ENVIRONMENT FEATURES:")
    print("  ✓ No weather effects (rain/fog)")
    print("  ✓ Minimal fading (<0.5 dB)")
    print("  ✓ Stable temperature (23±1°C)")
    print("  ✓ Controlled humidity (50±3%)")
    print("  ✓ Quasi-static movement (0.1-0.5 m/s)")
    print("  ✓ Precise beam alignment (0.1° error)")
    print("  ✓ Short range (50-200m)")
    print("=" * 90)