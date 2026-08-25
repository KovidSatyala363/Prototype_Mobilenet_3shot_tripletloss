import cv2
import numpy as np

# Disease-specific beta values (Bastiaans model)
DEFAULT_BETA_VALUES = {
    "unhealthy_brownspot_P": 3.0,
    "unhealthy_blackspot_G": 2.5,
    "default": 2.0
}

# Baseline CO2 assimilation rates (µmol CO2 m⁻² s⁻¹)
DEFAULT_BASELINE_ASSIMILATION = {
    "unhealthy_brownspot_P": 12.0,
    "unhealthy_blackspot_G": 10.0,
    "default": 11.0
}


def calculate_visible_leaf_damage(path, disease_type=None, min_damage=0.5):
    """Calculate visible leaf damage (%)."""

    img = cv2.imread(path)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Leaf mask: all leaf tissue (green + brown + black)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_brown = np.array([5, 40, 20])
    upper_brown = np.array([25, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 40])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_leaf = mask_green | mask_brown | mask_black

    kernel_small = np.ones((3, 3), np.uint8)
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_OPEN, kernel_small)
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_CLOSE, kernel_small)

    # Damage mask: diseased tissue + chlorotic halos
    lower_brown_spot = np.array([8, 60, 30])
    upper_brown_spot = np.array([22, 255, 180])
    mask_brown_spot = cv2.inRange(hsv, lower_brown_spot, upper_brown_spot)

    lower_black_spot = np.array([0, 0, 0])
    upper_black_spot = np.array([180, 255, 60])
    mask_black_spot = cv2.inRange(hsv, lower_black_spot, upper_black_spot)

    lower_chlorotic = np.array([18, 30, 40])
    upper_chlorotic = np.array([35, 255, 255])
    mask_chlorotic = cv2.inRange(hsv, lower_chlorotic, upper_chlorotic)

    if disease_type == "unhealthy_brownspot_P":
        mask_damage = mask_brown_spot | mask_chlorotic
    elif disease_type == "unhealthy_blackspot_G":
        mask_damage = mask_black_spot | mask_chlorotic
    else:
        mask_damage = mask_brown_spot | mask_black_spot | mask_chlorotic

    kernel = np.ones((3, 3), np.uint8)
    mask_damage = cv2.morphologyEx(mask_damage, cv2.MORPH_OPEN, kernel)
    mask_damage = cv2.morphologyEx(mask_damage, cv2.MORPH_CLOSE, kernel)
    mask_damage = cv2.bitwise_and(mask_damage, mask_leaf)

    total_leaf_pixels = cv2.countNonZero(mask_leaf)
    damaged_pixels = cv2.countNonZero(mask_damage)

    if total_leaf_pixels == 0:
        return 0.0

    damage_percent = (damaged_pixels / total_leaf_pixels) * 100.0

    if damage_percent == 0 and disease_type is not None and "unhealthy" in disease_type:
        return min_damage

    return damage_percent


def estimate_co2_assimilation_loss(damage_percent, disease_type=None,
                                   baseline_rate=None, beta=None):
    """Estimate CO2 assimilation loss using Bastiaans (1991) model."""

    if damage_percent is None or damage_percent < 0:
        return None

    x = min(damage_percent / 100.0, 1.0)

    if beta is None:
        beta = DEFAULT_BETA_VALUES.get(disease_type, DEFAULT_BETA_VALUES["default"])

    if baseline_rate is None:
        baseline_rate = DEFAULT_BASELINE_ASSIMILATION.get(
            disease_type, DEFAULT_BASELINE_ASSIMILATION["default"]
        )

    # Bastiaans non-linear model: Px/P0 = (1 - x)^β
    relative_remaining = (1.0 - x) ** beta
    relative_loss = 1.0 - relative_remaining

    absolute_loss = relative_loss * baseline_rate

    return round(absolute_loss, 3)


def estimate_disease_progression(current_damage_percent, disease_type=None,
                                 days=7, infection_rate=None):
    """Estimate future disease severity using logistic growth model."""

    if current_damage_percent is None or current_damage_percent <= 0:
        return None

    x0 = current_damage_percent / 100.0
    K = 1.0

    if infection_rate is None:
        if disease_type == "unhealthy_brownspot_P":
            infection_rate = 0.25
        elif disease_type == "unhealthy_blackspot_G":
            infection_rate = 0.15
        else:
            infection_rate = 0.15

    # Logistic growth: X(t) = K / (1 + ((K-x0)/x0) * exp(-r*K*t))
    projected_x = K / (1.0 + ((K - x0) / x0) * np.exp(-infection_rate * K * days))
    projected_percent = projected_x * 100.0

    return round(projected_percent, 2)


def estimate_days_to_severe_infection(damage_percent, disease_type=None,
                                      infection_rate=None, target=0.90):
    """Estimate days until leaf reaches severe infection (~90% damage)."""
    if damage_percent is None or damage_percent <= 0:
        return None

    x0 = damage_percent / 100.0
    K = 1.0

    if infection_rate is None:
        if disease_type == "unhealthy_brownspot_P":
            infection_rate = 0.25
        elif disease_type == "unhealthy_blackspot_G":
            infection_rate = 0.15
        else:
            infection_rate = 0.15

    if x0 >= target:
        return 0.0

    days = (1.0 / (infection_rate * K)) * np.log(((K - x0) / x0) * (target / (K - target)))
    return round(days, 1)
