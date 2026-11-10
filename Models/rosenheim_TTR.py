import numpy as np

# === CONSTANTS / COEFFICIENTS for Rosenheim function ====
# With levels of significance

BETA = 0.42
A_BUILDING = 0.78      # *
A_ELECTRICITY = 0.41   # ***
A_ROAD = 1.42          # ***
A_MACHINERY = 6.34     # ***
A_INVENTORY = 2.45     # *** (fresh food)

MAX_SEVERITY = 4
SEVERITY_THRESHOLDS = [34949.0000, 201979.0000, 875143.5000]

HISTORICAL_DAMAGES_THRESHOLDS = [21.0, 8636.0, 22435.0, 57166.0, 126935.8, 201979.0, 323941.2, 621929.2, 1177832.2, 2545856.2, 78133584.0]
PERCENTILES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]


# ===== PROXIES FUNCTIONS FOR ROSENHEIM FEATURES =========
# Note that for now these are quite arbitrary

def damage_to_severity(damage, thresholds = None) :
    """
    Maps a continuous damage metric into a discrete severity tier (1-4)
    """
    if thresholds is None:
        thresholds = SEVERITY_THRESHOLDS

    if damage in [1, 2, 3, 4] :
        return int(damage)

    t1, t2, t3 = thresholds
    if damage <= t1 :
        return 1
    elif damage <= t2 :
        return 2
    elif damage <= t3 :
        return 3
    return 4

def proxy_electricity_loss(event_duration_days, severity_index) :
    """
    Computes continuous days of electrical power outage
    Outage is bounded by hazard duration & scaled by severity ratio
    """
    severity_ratio = np.clip(severity_index / MAX_SEVERITY, 0.05, 1.0)
    elec_days = event_duration_days * severity_ratio
    return round(min(float(elec_days), 30.0), 2) # with a min to prevent outage the same duration as very big floods

def proxy_road_access() :
    """
    Binary blockage of road access
    In Rosenheim, 89% of facilities have transport issues
    """
    return 1 if np.random.rand() < 0.892 else 0

def proxy_building_damage(severity_index, p75_threshold = 0.75, p90_threshold = 0.90) :
    """
    Determines the binary moderate/severe building damage 
    In Rosenheim, only 7.8% of facilities have moderate/severe structural inundation
    This dummy is triggered when in upper percentile of severity
    """
    severity_ratio = np.clip(severity_index / MAX_SEVERITY, 0.0, 1.0)
    # probabilité calculées en proba conditionnelles: loi proba totales => somme donne ~7.8%
    if severity_ratio >= p90_threshold :
        proba_building = 0.45
    elif severity_ratio >= p75_threshold :
        proba_building = 0.15
    else :
        proba_building = 0.02
    return 1 if np.random.rand() < proba_building else 0

def proxy_machinery_damage(building_damage, severity_index) :
    """
    Binary severe equipment/machinery damage 
    In Rosenheim, severe machinery damage is rare: 1.7%
    We assume it occurs only if there's already building damage
    and under extreme severity of events
    """
    if (building_damage == 1) and (severity_index == 4) :
        proba_machinery = 0.30
    elif building_damage == 1 :
        proba_machinery = 0.09
    else :
        proba_machinery = 0.0
    return 1 if np.random.rand() < proba_machinery else 0

# ================== CAPACITY DEGRADATION =====================

def compute_scenario_capacity(nominal_capacity, days_closed, operating_days_year) :
    """
    Computes the effective annual capacity after accounting for disruption downtime
    By scaling the baseline capacity by the fraction of operational days
    """
    effective_days_closed = min(days_closed, operating_days_year)
    operational_fraction = max(0.0, 1.0 - effective_days_closed / operating_days_year)
    return round(nominal_capacity * operational_fraction, 2)

# ============ ROSENHEIM OLS REGRESSION ==================

def rosenheim_TTR(
    elec_loss_days = 0.0,
    limited_road_access = 0,
    building_damage = 0,
    machinery_damage = 0,
    inventory = 0,
    is_hit = True   ) :
    """
    Evaluates the Rosenheim et al (2024) OLS regression to estimate days closed 
    (accessibility => days closed) for an industrial facility
    - Dummy variables not relevant to the industrial context were set to 0
    - Non-significant coefficients are set to 0
    """

    if not is_hit :
        return 0.0

    days_closed = (
        0.42 
        + A_BUILDING * building_damage
        + A_ELECTRICITY * elec_loss_days
        + A_ROAD * limited_road_access
        + A_MACHINERY * machinery_damage
        + A_INVENTORY * inventory
    )
    return max (0.0, float(days_closed))

def evaluate_proxies(event_duration_days, severity_index) :
    """
    Maps hazard duration & severity to all proxies, then evaluates Rosenheim regression
    """

    severity_index = damage_to_severity(severity_index)

    elec_days = proxy_electricity_loss(event_duration_days, severity_index)
    damage_road = proxy_road_access(event_duration_days, severity_index)
    damage_building = proxy_building_damage(severity_index)
    damage_machinery = proxy_machinery_damage(damage_building, severity_index)

    days_closed = rosenheim_TTR(elec_days, damage_road, damage_building, damage_machinery)
    return {
        "days_closed_ttr" : round(float(days_closed), 2),
        "covariates": {
            "elec_loss_days": elec_days,
            "limited_road_access": damage_road,
            "building_damage": damage_building,
            "machinery_damage": damage_machinery,
        }
    }

# ============ KLIBI'S (2012) REGRESSION ==================

def klibi_ttr_depot(beta) :
    b = np.clip(beta, 0.0, 100.0)
    return 0.007 * (b ** 2) + 0.4709 * b

def klibi_ttr_ship_to_point(beta) :
    b = np.clip(beta, 0.0, 100.0)
    return 0.8419 * b

def map_severity_beta(severity) :
    """
    Maps discrete severity tiers to quartile midpoints for Beta
    """
    beta_map = {1: 12.5, 2: 37.5, 3: 62.5, 4: 87.5}
    beta = beta_map[severity]
    return beta

def map_continuous_damage_beta(damage) :
    """
    Associates a continuous damage value to a percentile (Beta)
    """
    beta = np.interp(damage, HISTORICAL_DAMAGES_THRESHOLDS, PERCENTILES)
    return float(np.clip(beta, 1.0, 99.0))

def evaluate_proxies_continuous_Klibi(event_duration_days, damage, facility_type = "depot", add_duration = False) :
    """
    Add hazard duration if we estimate that the facility is closed during the hazard + recovery time
    """
    beta = map_continuous_damage_beta(damage)
    tier = damage_to_severity(damage)

    if facility_type == "depot" :
        ttr_days = klibi_ttr_depot(beta)
    else :
        ttr_days = klibi_ttr_ship_to_point(beta)

    if add_duration :
        ttr_days += event_duration_days

    return {
        "days_closed_ttr" : round(float(ttr_days), 2),
        "covariates" : {
            "beta" : beta,
            "severity_level" : tier,
            "facility_type" : facility_type
        }
    }


# ================== MAIN =====================

def main() :
    np.random.seed(42)

    test_cases = [
        {"label": "1-day low-severity flood",  "dur": 1.0,  "sev": 1},
        {"label": "3-day moderate storm",       "dur": 3.0,  "sev": 2},
        {"label": "8-day severe riverine flood","dur": 8.0,  "sev": 3},
        {"label": "14-day extreme catastrophe", "dur": 14.0, "sev": 4}
    ]

    print(f"{'Scenario':<30} | {'ElecDays':<8} | {'Road':<5} | {'Bldg':<5} | {'Mach':<5} | {'Inv':<5} | {'TTR (Days)':<10}")
    print("-" * 80)

    for tc in test_cases:
        res = evaluate_proxies(
            event_duration_days=tc["dur"],
            severity_index=tc["sev"],
        )
        c = res["covariates"]
        print(f"{tc['label']:<30} | {c['elec_loss_days']:<8} | {c['limited_road_access']:<5} | "
            f"{c['building_damage']:<5} | {c['machinery_damage']:<5} | "
            f"{res['days_closed_ttr']:<10.2f}")

if __name__ == "__main__" :
    main()