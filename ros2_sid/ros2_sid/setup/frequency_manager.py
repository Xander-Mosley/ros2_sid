#!/usr/bin/env python3

"""
frequency_manager.py — Manage frequency configuration.

Features
--------
- View frequency configuration
- Edit frequency configuration
- Generate frequency arrays
- Manually specify frequency arrays
- Reset configuration to defaults
- Store configuration in a single JSON file

Author
------
Xander D. Mosley  
Email: XanderDMosley.Engineer@gmail.com  
Date: 8 Aug 2026
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path

# __all__ = []
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"

# ============================================================
# Constants / Paths
# ============================================================

SETUP_DIR = Path(__file__).resolve().parent
FREQUENCY_CONFIG_FILE = SETUP_DIR / "frequency_config.json"
CURRENT_AIRCRAFT_FILE = SETUP_DIR / "aircraft_library" / "current_aircraft.json"

SCREEN_WIDTH = 60
LABEL_WIDTH = 22

DEFAULT_FREQUENCY_CONFIG = {
    "maximum_frequency_hz": 1.5,
    "minimum_frequency_hz": 0.1,
    "frequency_step_hz": 0.04,
    "frequencies_hz": [0.10, 0.14, 0.18, 0.22, 0.26, 0.30,
                       0.34, 0.38, 0.42, 0.46, 0.50, 0.54,
                       0.58, 0.62, 0.66, 0.70, 0.74, 0.78,
                       0.82, 0.86, 0.90, 0.94, 0.98, 1.02,
                       1.06, 1.10, 1.14, 1.18, 1.22, 1.26,
                       1.30, 1.34, 1.38, 1.42, 1.46, 1.50],
    "alias_frequency_hz": 7.5,
    "sampling_frequency_hz": 37.5,
}

DEFAULT_WING_AREA_M2 = 15.0

SAMPLING_FREQUENCY_MULTIPLIER = 25.0
ALIAS_FREQUENCY_MULTIPLIER = 5.0

# =============================================================================
# Utility Functions
# =============================================================================

def pause():
    """Wait for the user before continuing."""
    input("\nPress Enter to continue...")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """
    Ask a yes/no question.
    Returns True for yes and False for no.
    If the user presses Enter, the default answer is returned.
    """
    default_text = "Y/n" if default else "y/N"

    while True:

        answer = input(f"{prompt} [{default_text}]: ").strip().lower()

        if answer == "":
            return default

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Please enter y or n.")


def input_value(prompt, current=None, allow_negative=True) -> float | None:
    """
    Prompt the user for a numeric value.
    If the user presses Enter without typing anything, the current value is returned.
    If the user enters an invalid value, they are prompted again.
    """
    label = f"{prompt}"
    while True:
        text = input(f"{label:<{LABEL_WIDTH}} : ").strip()

        if text == "":
            return current

        try:
            value = float(text)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if not allow_negative and value < 0:
            print("Value must be non-negative.")
            continue

        return value

# ============================================================
# File Functions
# ============================================================

def load_frequency_config() -> dict:
    """Load frequency configuration from frequency_config.json."""
    if not FREQUENCY_CONFIG_FILE.exists():
        config = deepcopy(DEFAULT_FREQUENCY_CONFIG)
        with FREQUENCY_CONFIG_FILE.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)
        return config

    try:
        with FREQUENCY_CONFIG_FILE.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError:
        print("Error: frequency_config.json contains invalid JSON.")
        raise
    except OSError as error:
        print(f"Error loading frequency configuration: {error}")
        raise

    return config


def save_frequency_config(config: dict) -> None:
    """Save frequency configuration to JSON."""
    try:
        with FREQUENCY_CONFIG_FILE.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)
    except OSError as error:
        print(f"Error saving frequency configuration: {error}")
        raise

# ============================================================
# Validation
# ============================================================

def validate_maximum_frequency(
    value: float, config: dict) -> tuple[bool, str]:
    """Validate maximum frequency."""
    if value <= 0:
        return False, "Maximum frequency must be greater than zero."
    if value <= config['minimum_frequency_hz']:
        return (
            False,
            "Maximum frequency must be greater than "
            "the minimum frequency.",
        )
    return True, ""


def validate_minimum_frequency(
    value: float, config: dict) -> tuple[bool, str]:
    """Validate minimum frequency."""
    if value <= 0:
        return False, "Minimum frequency must be greater than zero."
    if value >= config['maximum_frequency_hz']:
        return (
            False,
            "Minimum frequency must be less than "
            "the maximum frequency.",
        )
    return True, ""


def validate_frequency_step(
    value: float, config: dict) -> tuple[bool, str]:
    """Validate frequency step."""
    if value <= 0:
        return False, "Frequency step must be greater than zero."
    return True, ""


def validate_frequencies(
    frequencies: list[float],
    config: dict
    ) -> tuple[bool, str]:
    """Validate frequency array."""
    if not frequencies:
        return False, "At least one frequency must be provided."
    # if any(frequency <= config["minimum_frequency_hz"] for frequency in frequencies):
    #     return (
    #         False,
    #         "All frequencies must be greater than ."
    #         "the minimum frequency."
    #     )
    # if any(frequency <= config["maximum_frequency_hz"] for frequency in frequencies):
    #     return (
    #         False,
    #         "All frequencies must be less than ."
    #         "the maximum frequency."
    #     )
    return True, ""


def validate_alias_frequency(
    value: float, config: dict) -> tuple[bool, str]:
    """Validate alias frequency."""
    if value <= 0:
        return False, "Alias frequency must be greater than zero."
    return True, ""


def validate_sampling_frequency(
    value: float, config: dict) -> tuple[bool, str]:
    """Validate sampling frequency."""
    if value <= 0:
        return False, "Sampling frequency must be greater than zero."
    return True, ""


def validate_frequency_config(config: dict) -> bool:
    """Validate complete frequency configuration."""
    if not validate_maximum_frequency(
        config["maximum_frequency_hz"],
        config
        ):
        return False

    if not validate_minimum_frequency(
        config["minimum_frequency_hz"],
        config
        ):
        return False

    if not validate_frequency_step(
        config["frequency_step_hz"],
        config
        ):
        return False

    if not validate_frequencies(
        config["frequencies_hz"],
        config
        ):
        return False

    if not validate_alias_frequency(
        config["alias_frequency_hz"],
        config
        ):
        return False

    if not validate_sampling_frequency(
        config["sampling_frequency_hz"],
        config
        ):
        return False

    return True

# ============================================================
# Display
# ============================================================

def view_frequency_config(config: dict) -> None:
    """Display current frequency configuration."""
    print_page("CURRENT CONFIGURATION")

    print_section("Frequency Limits")
    print(f"{'Minimum Frequency':<{LABEL_WIDTH}} : {config['minimum_frequency_hz']:.3f} Hz")
    print(f"{'Maximum Frequency':<{LABEL_WIDTH}} : {config['maximum_frequency_hz']:.3f} Hz")
    print(f"{'Frequency Step':<{LABEL_WIDTH}} : {config['frequency_step_hz']:.3f} Hz")
    print()

    print_section("Frequency Array")
    frequencies = config["frequencies_hz"]
    print(f"{'Number of Frequencies':<{LABEL_WIDTH}} : {len(frequencies)}")
    if frequencies:
        print(f"{'Frequencies':<{LABEL_WIDTH}}")
        for frequency in frequencies:
            print(f"{'':<{LABEL_WIDTH}} : {frequency:.3f} hz")
    else:
        print(f"{'Frequencies':<{LABEL_WIDTH}} : None")
    print()

    print_section("Sampling")
    print(f"{'Sampling Frequency':<{LABEL_WIDTH}} : {config['sampling_frequency_hz']:.3f} Hz")
    print(f"{'Alias Frequency':<{LABEL_WIDTH}} : {config['alias_frequency_hz']:.3f} Hz")

# ============================================================
# Autofill
# ============================================================

def auto_frequency_config(config: dict) -> None:
    """Set frequency configuration based on current_aircraft.json"""
    print_page("Automatic Configuration")
    print("WARNING: This will replace the current frequency")
    print("configuration with the automatic configuration.")
    print()
    
    if not ask_yes_no(f"Continue?"):
        print("\nAutomatic configuration cancelled.")
        return
    print()

    try:
        with CURRENT_AIRCRAFT_FILE.open("r", encoding="utf-8") as file:
            aircraft = json.load(file)
    except FileNotFoundError:
        print("Error: current_aircraft.json was not found.")
        return
    except json.JSONDecodeError:
        print("Error: current_aircraft.json contains invalid JSON.")
        return
    except OSError as error:
        print(f"Error loading current_aircraft.json: {error}")
        return
    
    try:
        wing_area = aircraft["geometry"]["wing_area_m2"]
    except KeyError:
        print("Error: Wing area was not found in current_aircraft.json.")
        return
    if wing_area is None:
        print("Wing area has not been defined for the current aircraft.")
        print()
        print(
            "Automatic configuration cannot be scaled using "
            "the aircraft geometry."
        )
        print()
        if ask_yes_no("Use the default configuration?"):
            reset_frequency_config(config)
        else:
            print("\nAutomatic configuration cancelled.")
        return
    if wing_area <= 0:
        print("Error: Wing area must be greater than zero.")
        return
    
    scale_factor = math.sqrt(DEFAULT_WING_AREA_M2 / wing_area)

    minimum_frequency = (
        DEFAULT_FREQUENCY_CONFIG["minimum_frequency_hz"]
        * scale_factor
    )
    maximum_frequency = (
        DEFAULT_FREQUENCY_CONFIG["maximum_frequency_hz"]
        * scale_factor
    )
    frequency_step = (
        DEFAULT_FREQUENCY_CONFIG["frequency_step_hz"]
        * scale_factor
    )

    alias_frequency = (
        maximum_frequency
        * ALIAS_FREQUENCY_MULTIPLIER
    )
    sampling_frequency = (
        maximum_frequency
        * SAMPLING_FREQUENCY_MULTIPLIER
    )
    
    frequencies = []
    frequency = minimum_frequency
    while frequency <= maximum_frequency + (frequency_step * 1e-9):
        frequencies.append(round(frequency, 10))
        frequency += frequency_step
    
    config["minimum_frequency_hz"] = minimum_frequency
    config["maximum_frequency_hz"] = maximum_frequency
    config["frequency_step_hz"] = frequency_step
    config["frequencies_hz"] = frequencies
    config["alias_frequency_hz"] = alias_frequency
    config["sampling_frequency_hz"] = sampling_frequency

    save_frequency_config(config)
    print_page("Automatic Configuration Successful")

    print_section("Frequency Limits")
    print(f"{'Minimum Frequency':<{LABEL_WIDTH}} : {config['minimum_frequency_hz']:.3f} Hz")
    print(f"{'Maximum Frequency':<{LABEL_WIDTH}} : {config['maximum_frequency_hz']:.3f} Hz")
    print(f"{'Frequency Step':<{LABEL_WIDTH}} : {config['frequency_step_hz']:.3f} Hz")
    print()

    print_section("Frequency Array")
    frequencies = config["frequencies_hz"]
    print(f"{'Number of Frequencies':<{LABEL_WIDTH}} : {len(frequencies)}")
    if frequencies:
        print(f"{'Frequencies':<{LABEL_WIDTH}}")
        for frequency in frequencies:
            print(f"{'':<{LABEL_WIDTH}} : {frequency:.3f} hz")
    else:
        print(f"{'Frequencies':<{LABEL_WIDTH}} : None")
    print()

    print_section("Sampling")
    print(f"{'Sampling Frequency':<{LABEL_WIDTH}} : {config['sampling_frequency_hz']:.3f} Hz")
    print(f"{'Alias Frequency':<{LABEL_WIDTH}} : {config['alias_frequency_hz']:.3f} Hz")

# ============================================================
# Editing
# ============================================================

def edit_maximum_frequency(config: dict) -> None:
    """Edit maximum frequency."""
    current = config["maximum_frequency_hz"]
    print(f"{'Current maximum frequency':<{LABEL_WIDTH}}: {current:.3f} Hz")
    print()

    while True:
        frequency = input_value(
            "Enter new maximum frequency [Hz]",
            current=current,
            allow_negative=False
            )
        
        valid, message = validate_maximum_frequency(
            frequency,
            config,
        )

        if not valid:
            print(f"\n{message}")
            continue
        
        config["maximum_frequency_hz"] = frequency
        return


def edit_minimum_frequency(config: dict) -> None:
    """Edit minimum frequency."""
    current = config["minimum_frequency_hz"]
    print(f"{'Current minimum frequency':<{LABEL_WIDTH}}: {current:.3f} Hz")
    print()

    while True:
        frequency = input_value(
            "Enter new minimum frequency [Hz]",
            current=current,
            allow_negative=False
            )
        
        valid, message = validate_minimum_frequency(
            frequency,
            config,
        )

        if not valid:
            print(f"\n{message}")
            continue
        
        config["minimum_frequency_hz"] = frequency
        return


def edit_frequency_step(config: dict) -> None:
    """Edit frequency step."""
    current = config["frequency_step_hz"]
    print(f"{'Current frequency step':<{LABEL_WIDTH}}: {current:.3f} Hz")
    print()

    while True:
        step = input_value(
            "Enter new frequency step [Hz]",
            current=current,
            allow_negative=False
            )
        
        valid, message = validate_frequency_step(
            step,
            config,
        )

        if not valid:
            print(f"\n{message}")
            continue
        
        config["frequency_step_hz"] = step
        return


def edit_frequencies(config: dict) -> None:
    """Edit frequency array."""
    while True:
        print_page("Edit Frequencies")
        print(" 1. Generate from minimum, maximum, and step")
        print(" 2. Enter frequencies manually")
        print(" 0. Cancel")
        print()

        choice = input("Selection: ").strip()
        print()

        if choice == "1":
            minimum = config["minimum_frequency_hz"]
            maximum = config["maximum_frequency_hz"]
            step = config["frequency_step_hz"]

            frequencies = []
            frequency = minimum
            while frequency <= maximum + (step * 1e-9):
                frequencies.append(round(frequency, 10))
                frequency += step

            config["frequencies_hz"] = frequencies
            print(f"Generated {len(frequencies)} frequencies.")
            print(
                f"Range: {frequencies[0]:.3f} Hz "
                f"to {frequencies[-1]:.3f} Hz"
            )

            return

        elif choice == "2":
            print("Enter frequencies separated by commas.")
            print("Example: 0.1, 0.5, 1.0, 2.0, 5.0")
            print()

            value = input("Frequencies [Hz]: ").strip()

            try:
                frequencies = [
                    float(item.strip())
                    for item in value.split(",")
                    if item.strip()
                ]
            except ValueError:
                print("\nInvalid input. All values must be numeric.")
                pause()
                continue

            valid, message = validate_frequencies(
                frequencies,
                config,
            )

            if not valid:
                print(f"\n{message}")
                continue
            
            frequencies.sort()
            config["frequencies_hz"] = frequencies
            print(f"\nSet {len(frequencies)} frequencies.")
            return

        elif choice == "0":
            return
        
        else:
            print("Invalid selection.")
            pause()


def edit_alias_frequency(config: dict) -> None:
    """Edit alias frequency."""
    current = config["alias_frequency_hz"]
    print(f"{'Current alias frequency':<{LABEL_WIDTH}}: {current:.3f} Hz")
    print()

    while True:
        frequency = input_value(
            "Enter new maximum frequency [Hz]",
            current=current,
            allow_negative=False
            )
        
        valid, message = validate_alias_frequency(
            frequency,
            config,
        )

        if not valid:
            print(f"\n{message}")
            continue
        
        config["alias_frequency_hz"] = frequency
        return


def edit_sampling_frequency(config: dict) -> None:
    """Edit sampling frequency."""
    current = config["sampling_frequency_hz"]
    print(f"{'Current sampling frequency':<{LABEL_WIDTH}}: {current:.3f} Hz")
    print()

    while True:
        frequency = input_value(
            "Enter new maximum frequency [Hz]",
            current=current,
            allow_negative=False
            )
        
        valid, message = validate_sampling_frequency(
            frequency,
            config,
        )

        if not valid:
            print(f"\n{message}")
            continue
        
        config["sampling_frequency_hz"] = frequency
        return


def edit_frequency_config(config: dict) -> None:
    """Display frequency editing menu."""
    temp_config = deepcopy(config)
    modified = False

    while True:
        print_page("Edit Configuration")
        print(" 1. Save")
        print(" 2. Maximum Frequency")
        print(" 3. Minimum Frequency")
        print(" 4. Frequency Step")
        print(" 5. Frequencies")
        print(" 6. Alias Frequency")
        print(" 7. Sampling Frequency")
        print(" 0. Cancel")
        print()

        choice = input("Selection: ").strip()
        print()

        if choice == "1":
            if not modified:
                print("No changes to save.")
                pause()
                continue
            if validate_frequency_config(temp_config):
                config.clear()
                config.update(temp_config)
                save_frequency_config(config)
                print("Changes saved.")
                return
            pause()

        elif choice == "2":
            edit_maximum_frequency(temp_config)
            modified = True

        elif choice == "3":
            edit_minimum_frequency(temp_config)
            modified = True

        elif choice == "4":
            edit_frequency_step(temp_config)
            modified = True

        elif choice == "5":
            edit_frequencies(temp_config)
            modified = True
            pause()

        elif choice == "6":
            edit_alias_frequency(temp_config)
            modified = True

        elif choice == "7":
            edit_sampling_frequency(temp_config)
            modified = True

        elif choice == "0":
            if not modified:
                print("Nothing changed.")
                return
            if ask_yes_no("Discard unsaved changes?"):
                return

        else:
            print("Invalid selection.")
            pause()

# ============================================================
# Reset
# ============================================================

def reset_frequency_config(config: dict) -> None:
    """Reset frequency configuration to defaults."""
    print_page("Resetting Configuration")
    print("WARNING: This will replace the current frequency")
    print("configuration with the default configuration.")
    print()
    
    if not ask_yes_no(f"Continue?"):
        print("\nReset cancelled.")
        return

    config.clear()
    config.update(deepcopy(DEFAULT_FREQUENCY_CONFIG))

    save_frequency_config(config)

    return

# ============================================================
# Menus
# ============================================================

def clear_screen():
    """Clear the terminal screen."""
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        # Fall back to printing blank lines
        print("\n" * 100)


def print_header():
    print("=" * SCREEN_WIDTH)
    print("FREQUENCY MANAGER")
    print("=" * SCREEN_WIDTH)


def print_developer_info():
    print("Developed by Xander D. Mosley")
    print("-" * SCREEN_WIDTH)
    print()


def print_menu():
    print(" 1. View Current Configuration")
    print(" 2. Automatic Configuration")
    print(" 3. Edit Configuration")
    print(" 4. Reset Configuration")
    print(" 0. Quit")
    print()


def print_page(title: str):
    """Print a page header."""
    clear_screen()
    print_header()
    print(title)
    print("-" * SCREEN_WIDTH)
    print()


def print_section(title: str):
    """Print a section header."""
    print(title)
    print(("- " * ((len(title) + 1) // 2)).rstrip())


def main() -> None:
    config = load_frequency_config()

    while True:
        clear_screen()
        print_header()
        print_developer_info()
        print_menu()
        
        choice = input("Selection: ").strip()
        
        if choice == "1":
            view_frequency_config(config)

        elif choice == "2":
            auto_frequency_config(config)

        elif choice == "3":
            edit_frequency_config(config)

        elif choice == "4":
            reset_frequency_config(config)

        elif choice == "0":
            break

        else:
            print("\nInvalid selection.")

        pause()

# =============================================================================

if __name__ == "__main__":
    main()