import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy.stats import nakagami

class Vulnerable6GChannelSimulator:
    def __init__(self, frequency=30e9):
        """
        Simulator kanal 6G High-Fidelity dengan Vulnerability Analysis
        Mendukung dinamika mobilitas user dan fisika mmWave/Sub-THz (30-100+ GHz)
        """
        self.frequency = frequency
        freq_ghz = frequency / 1e9
        
        # --- FREQUENCY-DEPENDENT HARDWARE ---
        self.tx_power_dbm = 33
        # Antenna gain scales with frequency (beamforming improvement)
        self.tx_antenna_gain_dbi = 23 + 10 * np.log10(freq_ghz / 30.0)
        self.eirp_dbm = self.tx_power_dbm + self.tx_antenna_gain_dbi
        
        self.rx_gain = random.uniform(6, 10)
        # Noise figure worsens at higher frequencies
        self.rx_noise_figure = 6 + (freq_ghz - 30) * 0.03
        self.bandwidth = 400e6 if freq_ghz > 50 else 100e6
        self.c = 3e8
        self.wavelength = self.c / self.frequency
        
        # --- GEOMETRIC MOBILITY (User Position with Boundaries) ---
        start_dist = random.uniform(500, 1000)
        start_angle = random.uniform(0, 2*np.pi)
        self.user_x = start_dist * np.cos(start_angle)
        self.user_y = start_dist * np.sin(start_angle)
        self.distance = np.sqrt(self.user_x**2 + self.user_y**2)
        
        self.velocity = random.uniform(1, 5)  # m/s
        self.direction = random.uniform(0, 2*np.pi)
        
        # --- FREQUENCY-DEPENDENT ENVIRONMENT PHYSICS ---
        # Reflection loss increases with frequency (surface roughness effect)
        self.reflection_loss_per_bounce = 3.0 * np.sqrt(freq_ghz / 30.0)
        
        # Penetration loss scales dramatically with frequency
        self.wall_penetration_loss = 12.0 * (freq_ghz / 30.0)
        
        self.num_reflections = random.randint(2, 4)
        self.num_walls_penetrated = random.choice([0, 0, 0, 1])
        
        # --- PERSISTENT BLOCKAGE STATE (Markov Model) ---
        self.is_blocked = False
        self.blockage_timer = 0  # remaining time blocked (minutes)
        self.blockage_transition_prob = 0.02  # 2% chance per minute to enter blockage
        self.blockage_exit_prob = 0.3  # 30% chance per minute to exit blockage
        
        # --- ATMOSPHERIC & ENVIRONMENTAL ---
        self.temperature = 25.0
        self.humidity = 60.0
        self.water_vapor_density = self.calculate_water_vapor_density()
        self.fog_visibility = None
        
        # --- FADING STATE ---
        self.nakagami_m = 2.0
        self.rician_k = 8.0
        self.beam_misalignment_angle = 0.0
        
        # --- USERS & TRAFFIC ---
        self.num_users = random.randint(20, 50)
        self.traffic_density = random.uniform(0.6, 0.9)
        
    def calculate_water_vapor_density(self):
        """Water vapor density from temperature and humidity"""
        es = 6.112 * np.exp((17.67 * self.temperature) / (self.temperature + 243.5))
        e = (self.humidity / 100.0) * es
        rho_v = (e * 100) / (461.5 * (self.temperature + 273.15))
        return rho_v
    
    def get_rain_coefficients_frequency_dependent(self):
        """
        ITU-R P.838 coefficients scaled with frequency
        Rain becomes DEVASTATING at 100 GHz!
        """
        f = self.frequency / 1e9
        
        if f <= 40:
            return 0.187, 1.021  # 30 GHz
        elif f <= 70:
            # Interpolate
            t = (f - 40) / 30
            k = 0.187 + t * (0.65 - 0.187)
            alpha = 1.021 + t * (0.90 - 1.021)
            return k, alpha
        else:  # > 70 GHz - EXTREME sensitivity
            k = 1.1 + (f - 70) * 0.015
            alpha = 0.75
            return k, alpha
    
    def get_atmospheric_loss_db_km_frequency_dependent(self):
        """
        ITU-R P.676 with frequency dependence
        MASSIVE absorption at 60 GHz O2 peak!
        """
        f = self.frequency / 1e9
        
        # Base absorption (increases quadratically)
        base = 0.05 + (f / 100) ** 2 * 0.4
        
        # Oxygen absorption peak at 60 GHz (VERY STRONG!)
        o2_peak = 15.0 * np.exp(-((f - 60) ** 2) / 50.0)
        
        # Water vapor continuum (linear with freq & humidity)
        h2o_cont = 0.01 * f * (self.water_vapor_density / 7.5)
        
        return base + o2_peak + h2o_cont
    
    def update_position_and_blockage(self, interval_minutes=1):
        """
        MAIN UPDATE: Move user with boundary reflection & persistent blockage
        """
        # 1. UPDATE POSITION with boundary constraints (500-1000m)
        dist_move = self.velocity * (interval_minutes * 60)
        
        self.user_x += dist_move * np.cos(self.direction)
        self.user_y += dist_move * np.sin(self.direction)
        new_dist = np.sqrt(self.user_x ** 2 + self.user_y ** 2)
        
        # Boundary reflection: if outside [500, 1000]m, bounce back
        if new_dist > 1000 or new_dist < 500:
            # Reverse direction (180 degrees + random variation)
            self.direction += np.pi + random.uniform(-0.5, 0.5)
            # Push back into valid zone
            self.user_x += 2 * self.velocity * np.cos(self.direction)
            self.user_y += 2 * self.velocity * np.sin(self.direction)
            new_dist = np.sqrt(self.user_x ** 2 + self.user_y ** 2)
        
        self.distance = new_dist
        
        # Random walk for direction (natural meandering)
        self.direction += random.uniform(-0.3, 0.3)
        
        # 2. UPDATE BLOCKAGE STATE (Markov Chain)
        if self.is_blocked:
            # Currently blocked - try to exit
            if random.random() < self.blockage_exit_prob:
                self.is_blocked = False
                self.blockage_timer = 0
            else:
                self.blockage_timer += interval_minutes
        else:
            # Currently LOS - might enter blockage
            if random.random() < self.blockage_transition_prob:
                self.is_blocked = True
                self.blockage_timer = random.randint(2, 10)  # blocked for 2-10 min
    
    def calculate_path_loss_comprehensive(self, timestamp, is_raining, rain_rate):
        """
        Complete path loss with ALL effects - frequency dependent
        """
        # 1. FREE SPACE PATH LOSS (dynamic distance!)
        fspl = 20 * np.log10(self.distance) + 20 * np.log10(self.frequency) - 147.55
        
        # 2. ATMOSPHERIC ABSORPTION (frequency dependent, devastating at 60 GHz)
        atm_loss_db_km = self.get_atmospheric_loss_db_km_frequency_dependent()
        atm_loss = atm_loss_db_km * (self.distance / 1000.0)
        
        # 3. RAIN ATTENUATION (frequency dependent, EXTREME at 100 GHz!)
        rain_loss = 0
        if is_raining:
            k, alpha = self.get_rain_coefficients_frequency_dependent()
            gamma_rain = k * (rain_rate ** alpha)
            rain_loss = gamma_rain * (self.distance / 1000.0)
        
        # 4. FOG ATTENUATION (if present)
        fog_loss = 0
        if self.fog_visibility is not None and self.fog_visibility < 1000:
            f_ghz = self.frequency / 1e9
            M = 0.024 * ((self.fog_visibility / 1000) ** -1.05)
            gamma_fog = 0.4 * M * (f_ghz ** 2)
            fog_loss = gamma_fog * (self.distance / 1000)
        
        # 5. BLOCKAGE vs LOS (Persistent State)
        if self.is_blocked:
            # NLOS: Severe blockage + reflections
            # At 100 GHz, diffraction is negligible, blockage is MASSIVE
            freq_ghz = self.frequency / 1e9
            blockage_base = 25 + (freq_ghz - 30) * 0.5  # scales with freq
            blockage_loss = random.uniform(blockage_base, blockage_base + 15)
            
            reflection_loss = self.num_reflections * self.reflection_loss_per_bounce
            penetration_loss = self.num_walls_penetrated * self.wall_penetration_loss
            
            env_loss = blockage_loss + reflection_loss + penetration_loss
            los_status = False
            self.rician_k = random.uniform(0, 3)  # weak LOS component
            
        else:
            # LOS: Minimal extra loss, only light shadowing
            shadow_fading = np.random.normal(0, 3)
            env_loss = max(0, shadow_fading)
            los_status = True
            self.rician_k = random.uniform(8, 12)  # strong LOS
        
        # 6. BEAM MISALIGNMENT (more critical at higher frequencies)
        beam_loss = 0
        if los_status:  # Only relevant in LOS
            theta = np.abs(np.random.normal(0, 1.5))  # 1.5 degree std error
            beamwidth = 10.0 / np.sqrt(self.frequency / 30e9)  # narrower at high freq
            if theta < beamwidth:
                beam_loss = 12 * (theta / beamwidth) ** 2
            else:
                beam_loss = 12 + 10 * np.log10(theta / beamwidth)
            beam_loss = min(beam_loss, 15)
        
        # 7. SMALL-SCALE FADING (Nakagami)
        hour = timestamp.hour
        if los_status:
            m = random.uniform(2.5, 3.5) if (22 <= hour or hour < 6) else random.uniform(2.0, 3.0)
        else:
            m = random.uniform(1.0, 1.5)  # severe fading in NLOS
        
        self.nakagami_m = m
        fading_amplitude = nakagami.rvs(m, scale=np.sqrt(1 / m))
        small_scale_fading = 20 * np.log10(fading_amplitude)
        small_scale_fading = np.clip(small_scale_fading, -15, 3)
        
        # TOTAL PATH LOSS
        total_pl = (fspl + atm_loss + rain_loss + fog_loss + 
                    env_loss + beam_loss + small_scale_fading)
        
        return total_pl, {
            'fspl': fspl,
            'atm_loss': atm_loss,
            'rain_loss': rain_loss,
            'fog_loss': fog_loss,
            'env_loss': env_loss,
            'beam_loss': beam_loss,
            'small_scale_fading': small_scale_fading,
            'los_status': los_status
        }
    
    def calculate_interference(self, hour):
        """Time-varying interference (in dBm, NEGATIVE!)"""
        # Diurnal user activity
        if 22 <= hour or hour < 4:
            user_factor = random.uniform(0.4, 0.6)
            traffic_mult = random.uniform(0.2, 0.4)
        elif 17 <= hour < 22:
            user_factor = random.uniform(0.9, 1.2)
            traffic_mult = random.uniform(0.8, 1.0)
        else:
            user_factor = random.uniform(0.8, 1.0)
            traffic_mult = random.uniform(0.6, 0.8)
        
        active_users = int(self.num_users * user_factor)
        current_traffic = self.traffic_density * traffic_mult
        
        # Base interference (received power from interferers)
        interference_per_user = -105  # dBm
        
        if active_users > 0:
            interference_scale = 10 * np.log10(active_users * current_traffic / 10)
            base_interference = interference_per_user + interference_scale
        else:
            base_interference = -120
        
        adjacent_interference = random.uniform(-110, -100)
        
        total_interference = 10 * np.log10(
            10 ** (base_interference / 10) + 10 ** (adjacent_interference / 10)
        )
        
        total_interference = np.clip(total_interference, -120, -85)
        
        return total_interference, active_users, current_traffic
    
    def update_environment(self, timestamp):
        """Update temperature, humidity, etc."""
        hour = timestamp.hour
        temp_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        self.temperature = 25 + temp_variation + np.random.normal(0, 1)
        
        humidity_base = 70 - temp_variation * 2
        self.humidity = np.clip(humidity_base + np.random.normal(0, 5), 30, 95)
        self.water_vapor_density = self.calculate_water_vapor_density()
    
    def simulate_measurement(self, timestamp, is_raining=False, rain_rate=0):
        """Single measurement - COMPLETE"""
        hour = timestamp.hour
        
        # Update dynamics
        self.update_position_and_blockage(interval_minutes=1)
        self.update_environment(timestamp)
        
        # Path loss calculation
        path_loss, components = self.calculate_path_loss_comprehensive(
            timestamp, is_raining, rain_rate
        )
        
        # Interference
        interference, active_users, traffic = self.calculate_interference(hour)
        
        # LINK BUDGET
        rsrp = self.eirp_dbm + self.rx_gain - path_loss
        
        thermal_noise = -174 + 10 * np.log10(self.bandwidth)
        noise_power = thermal_noise + self.rx_noise_figure
        
        # SNR calculation
        inr = 10 * np.log10(10 ** (interference / 10) + 10 ** (noise_power / 10))
        snr = rsrp - inr
        sinr = snr  # same in this case
        
        rssi = 10 * np.log10(
            10 ** (rsrp / 10) + 10 ** (interference / 10) + 10 ** (noise_power / 10)
        )
        
        return {
            'timestamp': timestamp,
            'hour': hour,
            'frequency_ghz': round(self.frequency / 1e9, 1),
            
            # Position & Mobility
            'distance_m': round(self.distance, 2),
            'user_x': round(self.user_x, 2),
            'user_y': round(self.user_y, 2),
            'velocity_ms': round(self.velocity, 2),
            
            # Blockage State
            'is_blocked': self.is_blocked,
            'blockage_duration_min': self.blockage_timer,
            'los_status': components['los_status'],
            
            # Power Metrics
            'rsrp_dbm': round(rsrp, 2),
            'rssi_dbm': round(rssi, 2),
            'snr_db': round(snr, 2),
            'sinr_db': round(sinr, 2),
            
            # Path Loss Breakdown (VULNERABILITY ANALYSIS!)
            'path_loss_total_db': round(path_loss, 2),
            'fspl_db': round(components['fspl'], 2),
            'atmospheric_loss_db': round(components['atm_loss'], 2),
            'rain_loss_db': round(components['rain_loss'], 2),
            'fog_loss_db': round(components['fog_loss'], 2),
            'blockage_env_loss_db': round(components['env_loss'], 2),
            'beam_misalignment_loss_db': round(components['beam_loss'], 2),
            'small_scale_fading_db': round(components['small_scale_fading'], 2),
            
            # Environment
            'temperature_c': round(self.temperature, 2),
            'humidity_percent': round(self.humidity, 2),
            'is_raining': is_raining,
            'rain_rate_mm_h': rain_rate,
            'fog_visibility_m': self.fog_visibility if self.fog_visibility else 9999,
            
            # Interference
            'interference_dbm': round(interference, 2),
            'noise_power_dbm': round(noise_power, 2),
            'active_users': active_users,
            
            # Channel State
            'nakagami_m': round(self.nakagami_m, 2),
            'rician_k_db': round(self.rician_k, 2),
            
            # System Params
            'eirp_dbm': round(self.eirp_dbm, 2),
            'rx_gain_dbi': round(self.rx_gain, 2),
            'rx_noise_figure_db': round(self.rx_noise_figure, 2)
        }


def generate_weather_events(start_time, duration_days=7):
    """Generate realistic weather with intensity variation"""
    events = []
    
    # 3 rain events - varied intensity
    rain_intensities = [10, 25, 45]  # light, moderate, heavy
    for intensity in rain_intensities:
        day = random.randint(0, duration_days - 1)
        hour = random.choices(range(24), weights=[1]*6 + [2]*6 + [3]*6 + [2]*6)[0]
        
        rain_start = start_time + timedelta(days=day, hours=hour, minutes=random.randint(0, 59))
        rain_duration = random.randint(30, 180)
        
        events.append({
            'type': 'rain',
            'start': rain_start,
            'duration_minutes': rain_duration,
            'intensity': intensity + random.uniform(-5, 5)
        })
    
    # 2 fog events
    for _ in range(2):
        day = random.randint(0, duration_days - 1)
        hour = random.randint(4, 8)  # early morning
        
        fog_start = start_time + timedelta(days=day, hours=hour)
        fog_duration = random.randint(60, 240)
        visibility = random.uniform(100, 500)
        
        events.append({
            'type': 'fog',
            'start': fog_start,
            'duration_minutes': fog_duration,
            'intensity': visibility
        })
    
    return sorted(events, key=lambda x: x['start'])


def check_weather(timestamp, weather_events, simulator):
    """Check current weather conditions"""
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
    """
    MAIN SIMULATION: Compare 30 GHz vs 100 GHz vulnerability
    """
    print("=" * 90)
    print("  🔬 SIMULATOR KANAL 6G - VULNERABILITY & DYNAMICS ANALYSIS")
    print("=" * 90)
    
    start_time = datetime.now()
    
    # Generate weather (same for both frequencies)
    weather = generate_weather_events(start_time, duration_days=1)
    
    print("\n☁️ WEATHER EVENTS:")
    for i, event in enumerate(weather, 1):
        if event['type'] == 'rain':
            print(f"  {i}. 🌧️  Rain: {event['start'].strftime('%H:%M')} "
                    f"({event['duration_minutes']}min) - {event['intensity']:.1f} mm/h")
        else:
            print(f"  {i}. 🌫️  Fog:  {event['start'].strftime('%H:%M')} "
                    f"({event['duration_minutes']}min) - {event['intensity']:.0f}m visibility")
    
    # CREATE TWO SIMULATORS
    print("\n📡 INITIALIZING SIMULATORS...")
    sim_30 = Vulnerable6GChannelSimulator(frequency=30e9)
    sim_100 = Vulnerable6GChannelSimulator(frequency=100e9)
    
    # Force SAME initial position for fair comparison
    sim_100.user_x = sim_30.user_x
    sim_100.user_y = sim_30.user_y
    sim_100.distance = sim_30.distance
    sim_100.velocity = sim_30.velocity
    sim_100.direction = sim_30.direction
    
    print(f"  30 GHz:  EIRP={sim_30.eirp_dbm:.1f} dBm, NF={sim_30.rx_noise_figure:.1f} dB, BW={sim_30.bandwidth/1e6:.0f} MHz")
    print(f"  100 GHz: EIRP={sim_100.eirp_dbm:.1f} dBm, NF={sim_100.rx_noise_figure:.1f} dB, BW={sim_100.bandwidth/1e6:.0f} MHz")
    print(f"  Initial Distance: {sim_30.distance:.1f}m")
    
    print("\n⏳ RUNNING 24-HOUR SIMULATION...")
    
    results = []
    current_time = start_time
    
    # Simulate 24 hours (1440 minutes)
    for i in range(1440):
        is_raining, rain_rate = check_weather(current_time, weather, sim_30)
        
        # Simulate 30 GHz
        res_30 = sim_30.simulate_measurement(current_time, is_raining, rain_rate)
        
        # Sync position & blockage to 100 GHz (for fair comparison)
        sim_100.user_x = sim_30.user_x
        sim_100.user_y = sim_30.user_y
        sim_100.distance = sim_30.distance
        sim_100.is_blocked = sim_30.is_blocked
        sim_100.blockage_timer = sim_30.blockage_timer
        
        # Simulate 100 GHz
        res_100 = sim_100.simulate_measurement(current_time, is_raining, rain_rate)
        
        results.append({
            'time': current_time,
            'hour': current_time.hour,
            'distance': res_30['distance_m'],
            'is_blocked': res_30['is_blocked'],
            'is_raining': res_30['is_raining'],
            'rain_rate': rain_rate,
            
            # 30 GHz metrics
            'snr_30ghz': res_30['snr_db'],
            'rsrp_30ghz': res_30['rsrp_dbm'],
            'rain_loss_30ghz': res_30['rain_loss_db'],
            'atm_loss_30ghz': res_30['atmospheric_loss_db'],
            
            # 100 GHz metrics
            'snr_100ghz': res_100['snr_db'],
            'rsrp_100ghz': res_100['rsrp_dbm'],
            'rain_loss_100ghz': res_100['rain_loss_db'],
            'atm_loss_100ghz': res_100['atmospheric_loss_db'],
            
            # Vulnerability metrics
            'snr_degradation': res_30['snr_db'] - res_100['snr_db'],
            'outage_30ghz': res_30['snr_db'] < 0,
            'outage_100ghz': res_100['snr_db'] < 0
        })
        
        current_time += timedelta(minutes=1)
        
        if (i + 1) % 360 == 0:
            print(f"  ✓ {(i+1)//60} hours completed...")
    
    df = pd.DataFrame(results)
    
    # SAVE DATA
    filename = f'6g_vulnerability_analysis_{start_time.strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(filename, index=False)
    
    print("\n" + "=" * 90)
    print("✅ SIMULATION COMPLETE!")
    print("=" * 90)
    
    # VULNERABILITY ANALYSIS
    print("\n" + "=" * 90)
    print("📊 VULNERABILITY ANALYSIS - 30 GHz vs 100 GHz")
    print("=" * 90)
    
    # Overall statistics
    print("\n🔋 OVERALL PERFORMANCE:")
    print(f"{'Metric':<25} {'30 GHz':>15} {'100 GHz':>15} {'Difference':>15}")
    print("-" * 72)
    print(f"{'Avg SNR (dB)':<25} {df['snr_30ghz'].mean():>15.2f} {df['snr_100ghz'].mean():>15.2f} {(df['snr_30ghz'].mean() - df['snr_100ghz'].mean()):>15.2f}")
    print(f"{'Min SNR (dB)':<25} {df['snr_30ghz'].min():>15.2f} {df['snr_100ghz'].min():>15.2f} {(df['snr_30ghz'].min() - df['snr_100ghz'].min()):>15.2f}")
    print(f"{'Std Dev SNR (dB)':<25} {df['snr_30ghz'].std():>15.2f} {df['snr_100ghz'].std():>15.2f}")
    print(f"{'Outage Prob (%)':<25} {(df['outage_30ghz'].sum()/len(df)*100):>15.2f} {(df['outage_100ghz'].sum()/len(df)*100):>15.2f}")
    
    # Clear sky vs Rain vs Blockage
    clear = df[(df['rain_rate'] == 0) & (df['is_blocked'] == False)]
    rainy = df[df['rain_rate'] > 0]
    blocked = df[df['is_blocked'] == True]
    
    print("\n🌤️ CLEAR SKY CONDITIONS (LOS, No Rain):")
    if len(clear) > 0:
        print(f"  30 GHz:  SNR = {clear['snr_30ghz'].mean():.2f} ± {clear['snr_30ghz'].std():.2f} dB")
        print(f"  100 GHz: SNR = {clear['snr_100ghz'].mean():.2f} ± {clear['snr_100ghz'].std():.2f} dB")
        print(f"  → 100 GHz penalty: {clear['snr_30ghz'].mean() - clear['snr_100ghz'].mean():.2f} dB (higher FSPL)")
    
    print("\n🌧️ RAINY CONDITIONS:")
    if len(rainy) > 0:
        print(f"  30 GHz:  Avg SNR = {rainy['snr_30ghz'].mean():.2f} dB, Rain Loss = {rainy['rain_loss_30ghz'].mean():.2f} dB")
        print(f"  100 GHz: Avg SNR = {rainy['snr_100ghz'].mean():.2f} dB, Rain Loss = {rainy['rain_loss_100ghz'].mean():.2f} dB")
        
        rain_degradation_30 = clear['snr_30ghz'].mean() - rainy['snr_30ghz'].mean()
        rain_degradation_100 = clear['snr_100ghz'].mean() - rainy['snr_100ghz'].mean()
        
        print(f"  → Rain Degradation 30 GHz:  {rain_degradation_30:.2f} dB")
        print(f"  → Rain Degradation 100 GHz: {rain_degradation_100:.2f} dB ⚠️ SEVERE!")
        print(f"  → 100 GHz is {rain_degradation_100/rain_degradation_30:.1f}× more vulnerable to rain!")
    
    print("\n🏢 BLOCKAGE (NLOS) CONDITIONS:")
    if len(blocked) > 0:
        print(f"  30 GHz:  Avg SNR = {blocked['snr_30ghz'].mean():.2f} dB")
        print(f"  100 GHz: Avg SNR = {blocked['snr_100ghz'].mean():.2f} dB")
        
        blockage_outage_30 = (blocked['snr_30ghz'] < 0).sum() / len(blocked) * 100
        blockage_outage_100 = (blocked['snr_100ghz'] < 0).sum() / len(blocked) * 100
        
        print(f"  → Outage Rate 30 GHz:  {blockage_outage_30:.1f}%")
        print(f"  → Outage Rate 100 GHz: {blockage_outage_100:.1f}% ⚠️ CRITICAL!")
        print(f"  → 100 GHz suffers {blockage_outage_100/max(blockage_outage_30, 0.1):.1f}× more outages during blockage!")
    
    print("\n🌫️ ATMOSPHERIC ABSORPTION:")
    print(f"  30 GHz:  Avg = {df['atm_loss_30ghz'].mean():.3f} dB (negligible)")
    print(f"  100 GHz: Avg = {df['atm_loss_100ghz'].mean():.3f} dB")
    
    # Check if near 60 GHz O2 peak
    if 55 <= sim_100.frequency/1e9 <= 65:
        print(f"  → ⚠️ WARNING: Near 60 GHz O2 absorption peak! Severe attenuation!")
    
    print("\n📏 DISTANCE DYNAMICS:")
    print(f"  Range: {df['distance'].min():.1f}m - {df['distance'].max():.1f}m")
    print(f"  Avg:   {df['distance'].mean():.1f}m")
    print(f"  User stayed within boundary: {((df['distance'] >= 500) & (df['distance'] <= 1000)).all()}")
    
    print("\n🚫 BLOCKAGE STATISTICS:")
    blockage_rate = df['is_blocked'].sum() / len(df) * 100
    print(f"  Blockage Occurrence: {blockage_rate:.1f}% of time")
    
    # Consecutive blockage analysis
    blocked_periods = []
    current_block_len = 0
    for blocked in df['is_blocked']:
        if blocked:
            current_block_len += 1
        else:
            if current_block_len > 0:
                blocked_periods.append(current_block_len)
            current_block_len = 0
    
    if blocked_periods:
        print(f"  Avg Blockage Duration: {np.mean(blocked_periods):.1f} minutes")
        print(f"  Max Blockage Duration: {max(blocked_periods)} minutes")
        print(f"  Number of Blockage Events: {len(blocked_periods)}")
    
    print("\n💡 KEY INSIGHTS:")
    print(f"  1. SNR Degradation: 100 GHz is {df['snr_degradation'].mean():.1f} dB worse on average")
    print(f"  2. Rain Vulnerability: 100 GHz rain loss is {df['rain_loss_100ghz'].mean()/max(df['rain_loss_30ghz'].mean(), 0.01):.1f}× higher")
    print(f"  3. Blockage Sensitivity: 100 GHz nearly impossible during NLOS")
    print(f"  4. Atmospheric Effects: 100 GHz suffers {df['atm_loss_100ghz'].mean()/max(df['atm_loss_30ghz'].mean(), 0.01):.1f}× more atmospheric loss")
    
    print("\n🎯 RELIABILITY COMPARISON:")
    reliability_30 = (df['snr_30ghz'] >= 10).sum() / len(df) * 100
    reliability_100 = (df['snr_100ghz'] >= 10).sum() / len(df) * 100
    
    print(f"  SNR ≥ 10 dB (Good Service):")
    print(f"    30 GHz:  {reliability_30:.1f}% of time ✅")
    print(f"    100 GHz: {reliability_100:.1f}% of time")
    
    if reliability_100 < 50:
        print(f"    → ⚠️ 100 GHz unreliable for continuous service!")
    
    print("\n" + "=" * 90)
    print(f"📁 Data saved: {filename}")
    print("=" * 90)
    
    print("\n🧠 RECOMMENDATIONS FOR LSTM MODEL:")
    print("  • Use following features for prediction:")
    print("    - Temporal: hour, time_of_day")
    print("    - Environmental: rain_rate, fog, temperature, humidity")
    print("    - Channel: distance, is_blocked, los_status")
    print("    - Historical: previous SNR values (time-series)")
    print("  • Target: Predict SNR degradation events (SNR < 5 dB)")
    print("  • Label poor channel conditions for classification")
    print("  • Model will learn vulnerability patterns!")
    
    return df


def main_single_frequency(frequency_ghz=30, duration_days=7):
    """
    Single frequency simulation for detailed LSTM training data
    """
    print("=" * 90)
    print(f"  📡 6G CHANNEL SIMULATOR - {frequency_ghz} GHz")
    print("=" * 90)
    
    simulator = Vulnerable6GChannelSimulator(frequency=frequency_ghz * 1e9)
    start_time = datetime.now()
    weather = generate_weather_events(start_time, duration_days)
    
    print(f"\n📊 SYSTEM PARAMETERS:")
    print(f"  Frequency: {frequency_ghz} GHz")
    print(f"  EIRP: {simulator.eirp_dbm:.1f} dBm")
    print(f"  Rx Gain: {simulator.rx_gain:.1f} dBi")
    print(f"  Noise Figure: {simulator.rx_noise_figure:.1f} dB")
    print(f"  Bandwidth: {simulator.bandwidth/1e6:.0f} MHz")
    print(f"  Initial Distance: {simulator.distance:.1f}m")
    
    print(f"\n☁️ Weather Events: {len(weather)}")
    
    print("\n⏳ SIMULATING...")
    
    data = []
    current_time = start_time
    total_minutes = duration_days * 24 * 60
    
    for i in range(total_minutes):
        is_raining, rain_rate = check_weather(current_time, weather, simulator)
        measurement = simulator.simulate_measurement(current_time, is_raining, rain_rate)
        data.append(measurement)
        
        if (i + 1) % 1440 == 0:
            day = (i + 1) // 1440
            avg_snr = np.mean([d['snr_db'] for d in data[-1440:]])
            print(f"  ✓ Day {day}/{duration_days} - Avg SNR: {avg_snr:.2f} dB")
        
        current_time += timedelta(minutes=1)
    
    df = pd.DataFrame(data)
    filename = f'6g_channel_{frequency_ghz}ghz_{start_time.strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(filename, index=False)
    
    print("\n" + "=" * 90)
    print("✅ SIMULATION COMPLETE!")
    print("=" * 90)
    
    print(f"\n📈 STATISTICS:")
    print(f"  Total Samples: {len(df):,}")
    print(f"  SNR: {df['snr_db'].min():.1f} to {df['snr_db'].max():.1f} dB (μ={df['snr_db'].mean():.1f}, σ={df['snr_db'].std():.1f})")
    print(f"  Distance: {df['distance_m'].min():.1f} to {df['distance_m'].max():.1f}m")
    print(f"  Blockage Rate: {df['is_blocked'].sum()/len(df)*100:.1f}%")
    print(f"  Outage Rate (SNR<0): {(df['snr_db']<0).sum()/len(df)*100:.1f}%")
    
    print(f"\n🎯 QUALITY DISTRIBUTION:")
    excellent = (df['snr_db'] >= 20).sum()
    good = ((df['snr_db'] >= 10) & (df['snr_db'] < 20)).sum()
    fair = ((df['snr_db'] >= 0) & (df['snr_db'] < 10)).sum()
    poor = (df['snr_db'] < 0).sum()
    
    print(f"  Excellent (≥20 dB): {excellent:6d} ({excellent/len(df)*100:5.1f}%)")
    print(f"  Good (10-20 dB):    {good:6d} ({good/len(df)*100:5.1f}%)")
    print(f"  Fair (0-10 dB):     {fair:6d} ({fair/len(df)*100:5.1f}%)")
    print(f"  Poor (<0 dB):       {poor:6d} ({poor/len(df)*100:5.1f}%)")
    
    print(f"\n📁 File saved: {filename}")
    print("\n💡 Data ready for LSTM training!")
    
    return df


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 90)
    print("  6G CHANNEL SIMULATOR WITH VULNERABILITY ANALYSIS")
    print("=" * 90)
    print("\nChoose simulation mode:")
    print("  1. Vulnerability Analysis (30 GHz vs 100 GHz comparison - 24 hours)")
    print("  2. Single Frequency Detailed Simulation (for LSTM training - 7 days)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🔬 Running Vulnerability Analysis...")
        df = main_vulnerability_analysis()
        
    elif choice == "2":
        freq = input("Enter frequency in GHz (e.g., 30, 60, 100): ").strip()
        try:
            freq_val = float(freq)
            if freq_val < 10 or freq_val > 300:
                print("⚠️ Frequency out of range. Using 30 GHz.")
                freq_val = 30
        except:
            print("⚠️ Invalid input. Using 30 GHz.")
            freq_val = 30
        
        days = input("Enter simulation duration in days (1-30, default 7): ").strip()
        try:
            days_val = int(days)
            if days_val < 1 or days_val > 30:
                days_val = 7
        except:
            days_val = 7
        
        print(f"\n📡 Running {freq_val} GHz simulation for {days_val} days...")
        df = main_single_frequency(frequency_ghz=freq_val, duration_days=days_val)
        
    else:
        print("\n⚠️ Invalid choice. Running default vulnerability analysis...")
        df = main_vulnerability_analysis()
    
    print("\n✨ Simulation completed successfully!")
    print("=" * 90)