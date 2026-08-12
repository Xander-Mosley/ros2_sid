#!/usr/bin/env python3

"""
aircraft_manager.py — Manage the aircraft library.

Features
--------
- Select current aircraft
- View aircraft information
- Create aircraft definitions
- Edit aircraft definitions
- Delete aircraft
- Store all aircraft as JSON files
- Convert all user inputs to SI units

Author
------
Xander D. Mosley  
Email: XanderDMosley.Engineer@gmail.com  
Date: 22 Jul 2026
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil

# __all__ = []
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"

# =============================================================================
# Constants / Paths
# =============================================================================

SETUP_DIR = Path(__file__).resolve().parent
AIRCRAFT_LIBRARY = SETUP_DIR / "aircraft_library"
CURRENT_AIRCRAFT = AIRCRAFT_LIBRARY / "current_aircraft.json"

LIBRARY_DIR = SETUP_DIR.parent
USER_SETTINGS = LIBRARY_DIR / "user_settings.json"

SCREEN_WIDTH = 60
LABEL_WIDTH = 22

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# =============================================================================
# Unit Conversion Constants
# =============================================================================

LB_TO_KG = 0.45359237
FT_TO_M = 0.3048
IN_TO_M = 0.0254
FT2_TO_M2 = 0.09290304
SLUGFT2_TO_KGM2 = 1.35581795

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class Metadata:
    name: str = ""
    description: str = ""
    created: str = ""
    modified: str = ""

@dataclass
class Geometry:
    mass_kg: float | None = None
    wing_span_m: float | None = None
    wing_area_m2: float | None = None
    mac_m: float | None = None

@dataclass
class Inertia:
    Ixx_kgm2: float | None = None
    Iyy_kgm2: float | None = None
    Izz_kgm2: float | None = None
    
    Ixy_kgm2: float | None = None
    Ixz_kgm2: float | None = None
    Iyz_kgm2: float | None = None

@dataclass
class LongitudinalDerivatives:
    alpha: float | None = None
    q: float | None = None
    de: float | None = None
    alpha_dot: float | None = None
    w_dot: float | None = None
    inertia_derivative1: float | None = None
    inertia_derivative2: float | None = None
    thrust_derivative: float | None = None

@dataclass
class LateralDerivatives:
    beta: float | None = None
    p: float | None = None
    r: float | None = None
    da: float | None = None
    dr: float | None = None
    beta_dot: float | None = None
    inertia_derivative1: float | None = None
    inertia_derivative2: float | None = None
    thrust_derivative: float | None = None

@dataclass
class LongitudinalCoefficients:
    CX0: float | None = None
    CZ0: float | None = None
    Cm0: float | None = None

    CX: LongitudinalDerivatives = field(default_factory=LongitudinalDerivatives)
    CZ: LongitudinalDerivatives = field(default_factory=LongitudinalDerivatives)
    Cm: LongitudinalDerivatives = field(default_factory=LongitudinalDerivatives)

@dataclass
class LateralCoefficients:
    CY0: float | None = None
    Cl0: float | None = None
    Cn0: float | None = None

    CY: LateralDerivatives = field(default_factory=LateralDerivatives)
    Cl: LateralDerivatives = field(default_factory=LateralDerivatives)
    Cn: LateralDerivatives = field(default_factory=LateralDerivatives)

@dataclass
class AeroCoefficients:
    longitudinal: LongitudinalCoefficients = field(default_factory=LongitudinalCoefficients)
    lateral: LateralCoefficients = field(default_factory=LateralCoefficients)

@dataclass
class Aircraft:
    metadata: Metadata
    geometry: Geometry
    inertia: Inertia
    aerodynamics: AeroCoefficients

# =============================================================================
# Utility Functions
# =============================================================================

def pause():
    """Wait for the user before continuing."""
    input("\nPress Enter to continue...")


def choose_aircraft(title: str) -> Aircraft | None:
    """Prompt the user to select an aircraft."""
    print_page(title)

    aircraft_files = print_aircraft_list()
    if not aircraft_files:
        return None

    while True:
        choice = input("\nSelect aircraft (0 to cancel): ").strip()

        try:
            choice = int(choice)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if choice == 0:
            return None
        if 1 <= choice <= len(aircraft_files):
            return load_aircraft(aircraft_files[choice - 1].stem)

        print("Invalid selection.")


def format_value(value: float | None,
                 units: str = "",
                 precision: int = 3) -> str:
    """Format a numeric value or return 'Unknown'."""
    if value is None:
        return "Unknown"
    return f"{value:.{precision}f} {units}".rstrip()

def format_mass(value_kg: float | None, units: str) -> str:
    """Format a mass in the user's preferred units."""
    if value_kg is None:
        return "Unknown"
    if units == "metric":
        return f"{value_kg:.3f} kg"
    return f"{value_kg / LB_TO_KG:.3f} lb"

def format_length(value_m: float | None, units: str) -> str:
    """Format a length using the preferred unit system."""
    if value_m is None:
        return "Unknown"
    if units == "metric":
        if value_m < 1.0:
            return f"{value_m * 100:.1f} cm"
        return f"{value_m:.3f} m"
    else:  # imperial
        value_ft = value_m / FT_TO_M
        if value_ft < 1.0:
            return f"{value_m / IN_TO_M:.2f} in"
        return f"{value_ft:.3f} ft"

def format_area(value_m2: float | None, units: str) -> str:
    """Format an area in the user's preferred units."""
    if value_m2 is None:
        return "Unknown"
    if units == "metric":
        return f"{value_m2:.3f} m²"
    return f"{value_m2 / FT2_TO_M2:.3f} ft²"

def format_inertia(value_kgm2: float | None, units: str) -> str:
    """Format a moment of inertia in the user's preferred units."""
    if value_kgm2 is None:
        return "Unknown"
    if units == "metric":
        return f"{value_kgm2:.3f} kg·m²"
    return f"{value_kgm2 / SLUGFT2_TO_KGM2:.3f} slug·ft²"


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


def timestamp():
    """Return ISO timestamp."""
    return datetime.now().isoformat(timespec="seconds")

# =============================================================================
# File Functions
# =============================================================================

def load_user_settings() -> dict:
    """Load the global user settings."""
    defaults = {
        "preferred_units": "metric",
    }

    if not USER_SETTINGS.exists():
        return defaults

    try:
        with open(USER_SETTINGS, "r", encoding="utf-8") as f:
            settings = json.load(f)
    except (OSError, json.JSONDecodeError):
        print("Warning: Unable to read user_settings.json.")
        print("Using default settings.")
        return defaults.copy()

    defaults.update(settings)
    return defaults


def load_aircraft(name: str) -> Aircraft:
    """
    Load an aircraft object from the aircraft library.
    Takes the aircraft name (without .json) as input.
    """
    filename = AIRCRAFT_LIBRARY / f"{name}.json"
    if not filename.exists():
        raise FileNotFoundError(f"Aircraft '{name}' does not exist.")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in aircraft file '{name}'."
        ) from e

    aero = data["aerodynamics"]
    longitudinal = aero["longitudinal"]
    lateral = aero["lateral"]

    try:
        return Aircraft(
            metadata=Metadata(**data["metadata"]),

            geometry=Geometry(**data["geometry"]),
            inertia=Inertia(**data["inertia"]),

            aerodynamics=AeroCoefficients(
                longitudinal=LongitudinalCoefficients(
                    CX0=longitudinal["CX0"],
                    CZ0=longitudinal["CZ0"],
                    Cm0=longitudinal["Cm0"],

                    CX=LongitudinalDerivatives(**longitudinal["CX"]),
                    CZ=LongitudinalDerivatives(**longitudinal["CZ"]),
                    Cm=LongitudinalDerivatives(**longitudinal["Cm"]),
                ),

                lateral=LateralCoefficients(
                    CY0=lateral["CY0"],
                    Cl0=lateral["Cl0"],
                    Cn0=lateral["Cn0"],

                    CY=LateralDerivatives(**lateral["CY"]),
                    Cl=LateralDerivatives(**lateral["Cl"]),
                    Cn=LateralDerivatives(**lateral["Cn"]),
                ),
            ),
        )
    except KeyError as e:
        raise ValueError(
            f"Aircraft file '{name}' is missing required field '{e.args[0]}'."
        ) from e


def get_aircraft_list(include_current: bool = False) -> list[Path]:
    """
    Return all aircraft JSON files sorted by aircraft name.
    Exclude the current aircraft if include_current is False.
    """
    aircraft = sorted(AIRCRAFT_LIBRARY.glob("*.json"))
    if not include_current:
        aircraft = [f for f in aircraft if f.name != CURRENT_AIRCRAFT.name]
    return aircraft

def get_current_aircraft_name() -> str | None:
    """Return the name of the selected aircraft."""
    if not CURRENT_AIRCRAFT.exists():
        return None
    try:
        return load_aircraft("current_aircraft").metadata.name
    except (FileNotFoundError, OSError, ValueError):
        return None

def print_aircraft_list() -> None:
    """Print a list of aircraft names and a selection indicator."""
    aircraft_files = get_aircraft_list()
    if not aircraft_files:
        print("No aircraft found.")
        return []
    current = get_current_aircraft_name()
    for i, file in enumerate(aircraft_files, start=1):
        selected_marker = "\t\t<current>" if file.stem == current else ""
        print(f"{i:2d}. {file.stem}{selected_marker}")
    return aircraft_files


def select_aircraft(aircraft: Aircraft) -> bool:
    """
    Set the specified aircraft object as the current aircraft.
    Returns True if successful, False otherwise.
    """
    current = get_current_aircraft_name()
    if current == aircraft.metadata.name:
        print(f"\n'{current}' is already the current aircraft.")
        return True

    source = AIRCRAFT_LIBRARY / f"{aircraft.metadata.name}.json"
    if not source.exists():
        print(f"\nAircraft '{aircraft.metadata.name}' does not exist.")
        return False

    try:
        shutil.copy2(source, CURRENT_AIRCRAFT)
        print(f"\nCurrent aircraft set to '{aircraft.metadata.name}'.")
        return True
    except OSError as e:
        print(f"\nFailed to select aircraft.")
        print(e)
        return False


def validate_aircraft_name(name: str) -> tuple[bool, str]:
    """
    Validate an aircraft name before using it as a filename.
    Returns a tuple (is_valid, reason). If valid, reason is an empty string.
    """
    if not name:
        return False, "Name cannot be empty."

    # Windows invalid characters
    invalid_chars = r'\/:*?"<>|'

    if re.search(f"[{re.escape(invalid_chars)}]", name):
        return False, (
            "Name contains invalid characters "
            r'(\ / : * ? " < > |).'
        )

    # Prevent hidden/path-like names
    if name in (".", ".."):
        return False, "Invalid name."

    # Prevent current_aircraft name.
    if name.lower() == "current_aircraft":
        return False, "Name cannot be 'current_aircraft'."

    # Prevent names that end with .json
    if name.lower().endswith(".json"):
        return False, "Do not include the .json extension."

    # Prevent Windows reserved names
    if name.upper() in WINDOWS_RESERVED_NAMES:
        return False, "Reserved system name."

    # Prevent leading/trailing spaces
    if name.strip() != name:
        return False, "Name cannot start or end with spaces."

    # Windows does not allow trailing periods/spaces
    if name[-1] in (" ", "."):
        return False, (
            "Name cannot end with a space or period."
        )

    return True, ""


def save_aircraft(
    aircraft: Aircraft,
    overwrite: bool = True,
    original_name: str | None = None) -> bool:
    """
    Save an aircraft definition.

    If the aircraft has been renamed, the old file is removed only after
    the new file has been written successfully. The save is performed
    atomically using a temporary file to avoid data corruption.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft object to save.
    overwrite : bool, default=True
        Whether an existing file may be overwritten.
    original_name : str | None, default=None
        Original aircraft name before editing. Used to safely rename the
        aircraft file if the name has changed.

    Returns
    -------
    bool
        True if the save completed successfully.
    """
    valid, reason = validate_aircraft_name(aircraft.metadata.name)
    if not valid:
        print(f"\nCannot save aircraft: {reason}")
        return False

    filename = AIRCRAFT_LIBRARY / (f"{aircraft.metadata.name}.json")
    temp_filename = filename.with_suffix(".json.tmp")
    
    old_filename = None
    renamed = (
        original_name is not None
        and original_name != aircraft.metadata.name
        )
    if original_name is not None:
        old_filename = AIRCRAFT_LIBRARY / f"{original_name}.json"
    
    # Prevent accidental overwrite when renaming
    if renamed and filename.exists():
        print(f"\nAircraft '{aircraft.metadata.name}' already exists.")
        return False
    # Prevent overwrite when creating a new aircraft
    if not renamed and filename.exists() and not overwrite:
        print(f"\nAircraft '{aircraft.metadata.name}' already exists.")
        return False

    try:
        aircraft.metadata.modified = timestamp()
        # Write to temporary file first
        with open(temp_filename, "w", encoding="utf-8") as f:
            json.dump(
                asdict(aircraft),
                f,
                indent=4,
                sort_keys=False,
            )
        # Atomically replace/create the destination
        temp_filename.replace(filename)
        # Remove the old file after the new one exists
        if renamed and old_filename and old_filename.exists():
            old_filename.unlink()
        # Keep current.json synchronized
        if renamed:
            current = get_current_aircraft_name()
            if current == original_name:
                select_aircraft(aircraft)
        return True

    except (OSError, TypeError) as e:
        # Remove leftover temporary file if present
        try:
            if temp_filename.exists():
                temp_filename.unlink()
        except OSError:
            pass
        print("\nFailed to save aircraft.")
        print(e)
        return False

# =============================================================================
# View Functions
# =============================================================================

def print_metadata(metadata: Metadata) -> None:
    print_section("Metadata")
    print(f"{'Created':<{LABEL_WIDTH}} : {metadata.created}")
    print(f"{'Modified':<{LABEL_WIDTH}} : {metadata.modified}")
    print(f"{'Description':<{LABEL_WIDTH}} : {metadata.description}")
    print()

def print_geometry(geometry: Geometry, units: str) -> None:
    print_section("Geometry")
    print(f"{'Mass':<{LABEL_WIDTH}} : {format_mass(geometry.mass_kg, units)}")
    print(f"{'Wing Span':<{LABEL_WIDTH}} : {format_length(geometry.wing_span_m, units)}")
    print(f"{'Wing Area':<{LABEL_WIDTH}} : {format_area(geometry.wing_area_m2, units)}")
    print(f"{'MAC':<{LABEL_WIDTH}} : {format_length(geometry.mac_m, units)}")
    print()

def print_inertia(inertia: Inertia, units: str) -> None:
    print_section("Moments of Inertia")
    print(f"{'Ixx':<{LABEL_WIDTH}} : {format_inertia(inertia.Ixx_kgm2, units)}")
    print(f"{'Iyy':<{LABEL_WIDTH}} : {format_inertia(inertia.Iyy_kgm2, units)}")
    print(f"{'Izz':<{LABEL_WIDTH}} : {format_inertia(inertia.Izz_kgm2, units)}")
    print(f"{'Ixy':<{LABEL_WIDTH}} : {format_inertia(inertia.Ixy_kgm2, units)}")
    print(f"{'Ixz':<{LABEL_WIDTH}} : {format_inertia(inertia.Ixz_kgm2, units)}")
    print(f"{'Iyz':<{LABEL_WIDTH}} : {format_inertia(inertia.Iyz_kgm2, units)}")
    print()

def print_longitudinal_coefficients(longitudinal: LongitudinalCoefficients) -> None:
    print_section("Longitudinal Trim")
    print(f"{'CX0':<{LABEL_WIDTH}} : {format_value(longitudinal.CX0)}")
    print(f"{'CZ0':<{LABEL_WIDTH}} : {format_value(longitudinal.CZ0)}")
    print(f"{'Cm0':<{LABEL_WIDTH}} : {format_value(longitudinal.Cm0)}")
    print()

def print_lateral_coefficients(lateral: LateralCoefficients) -> None:
    print_section("Lateral Trim")
    print(f"{'CY0':<{LABEL_WIDTH}} : {format_value(lateral.CY0)}")
    print(f"{'Cl0':<{LABEL_WIDTH}} : {format_value(lateral.Cl0)}")
    print(f"{'Cn0':<{LABEL_WIDTH}} : {format_value(lateral.Cn0)}")
    print()

def print_aircraft(aircraft: Aircraft, units: str) -> None:
    """Print all details of an aircraft."""
    print_page(f"Aircraft: {aircraft.metadata.name}")

    print_metadata(aircraft.metadata)
    print_geometry(aircraft.geometry, units)
    print_inertia(aircraft.inertia, units)
    print_longitudinal_coefficients(aircraft.aerodynamics.longitudinal)
    print_lateral_coefficients(aircraft.aerodynamics.lateral)

# =============================================================================
# Input Functions
# =============================================================================

def get_aircraft_name() -> str:
    """Prompt for a unique aircraft name."""
    while True:
        name = input("Aircraft name: ").strip()
        valid, reason = validate_aircraft_name(name)
        if not valid:
            print(reason)
            continue

        filename = AIRCRAFT_LIBRARY / f"{name}.json"
        if filename.exists():
            print("An aircraft with this name already exists.")
            continue

        return name

def input_multiline(prompt: str) -> str:
    """
    Read a multiline string.
    Finish by pressing Enter on an empty line.
    """
    print(prompt)
    print("(Press Enter on a blank line to finish.)")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

def edit_metadata(metadata: Metadata | None = None) -> Metadata:
    """
    Create or edit aircraft metadata object.
    If None is provided, a new object is created.
    Returns the updated metadata object.
    """
    if metadata is None:
        metadata = Metadata()

    print_section("Metadata")

    if metadata.name:
        original_name = metadata.name

        while True:
            value = input(f"Name [{metadata.name}]: ").strip()

            # Keep the existing name
            if value == "":
                break

            valid, reason = validate_aircraft_name(value)
            if not valid:
                print(reason)
                continue

            filename = AIRCRAFT_LIBRARY / f"{value}.json"

            if value != original_name and filename.exists():
                print("An aircraft with this name already exists.")
                continue

            metadata.name = value
            break
    else:
        metadata.name = get_aircraft_name()

    if metadata.description:
        print("\nCurrent description:")
        print(metadata.description)

    if ask_yes_no("Would you like to enter a new description?"):
        metadata.description = input_multiline("Enter aircraft description: ")

    return metadata


def input_value(prompt, current=None, allow_negative=True) -> float | None:
    """
    Prompt the user for a numeric value.
    If the user presses Enter without typing anything, the current value is returned.
    If the user enters an invalid value, they are prompted again.
    """
    label = f"{prompt}"
    while True:
        if current is None:
            text = input(f"{label:<{LABEL_WIDTH}} : ")
        else:
            text = input(f"{f'{label} - ({current})':<{LABEL_WIDTH}} : ")
        text = text.strip()

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

def edit_geometry(units: str, geometry: Geometry = None) -> Geometry:
    print_section("Geometry")

    if geometry is None:
        geometry = Geometry()

    if units == "metric":
        mass = input_value("Mass [kg]", geometry.mass_kg, allow_negative=False)
        span = input_value("Wing span [m]", geometry.wing_span_m, allow_negative=False)
        area = input_value("Wing area [m²]", geometry.wing_area_m2, allow_negative=False)
        mac = input_value("MAC [m]", geometry.mac_m, allow_negative=False)
    else:
        mass = input_value("Mass [lb]", geometry.mass_kg / LB_TO_KG if geometry.mass_kg is not None else None, allow_negative=False)
        span = input_value("Wing span [ft]", geometry.wing_span_m / FT_TO_M if geometry.wing_span_m is not None else None, allow_negative=False)
        area = input_value("Wing area [ft²]", geometry.wing_area_m2 / FT2_TO_M2 if geometry.wing_area_m2 is not None else None, allow_negative=False)
        mac = input_value("MAC [in]", geometry.mac_m / IN_TO_M if geometry.mac_m is not None else None, allow_negative=False)

        if mass is not None:
            mass *= LB_TO_KG
        if span is not None:
            span *= FT_TO_M
        if area is not None:
            area *= FT2_TO_M2
        if mac is not None:
            mac *= IN_TO_M

    return Geometry(
        mass_kg=mass,
        wing_span_m=span,
        wing_area_m2=area,
        mac_m=mac,
    )

def edit_inertia(units: str, inertia: Inertia = None) -> Inertia:
    print_section("Moments of Inertia")

    if inertia is None:
        inertia = Inertia()

    if units == "metric":
        Ixx = input_value("Ixx [kg·m²]", inertia.Ixx_kgm2, allow_negative=False)
        Iyy = input_value("Iyy [kg·m²]", inertia.Iyy_kgm2, allow_negative=False)
        Izz = input_value("Izz [kg·m²]", inertia.Izz_kgm2, allow_negative=False)
        Ixy = input_value("Ixy [kg·m²]", inertia.Ixy_kgm2)  # Likely has symmetry, but allow user to enter if known
        Ixz = input_value("Ixz [kg·m²]", inertia.Ixz_kgm2)
        Iyz = input_value("Iyz [kg·m²]", inertia.Iyz_kgm2)  # Likely has symmetry, but allow user to enter if known
    else:
        Ixx = input_value("Ixx [slug·ft²]", inertia.Ixx_kgm2 / SLUGFT2_TO_KGM2 if inertia.Ixx_kgm2 is not None else None, allow_negative=False)
        Iyy = input_value("Iyy [slug·ft²]", inertia.Iyy_kgm2 / SLUGFT2_TO_KGM2 if inertia.Iyy_kgm2 is not None else None, allow_negative=False)
        Izz = input_value("Izz [slug·ft²]", inertia.Izz_kgm2 / SLUGFT2_TO_KGM2 if inertia.Izz_kgm2 is not None else None, allow_negative=False)
        Ixy = input_value("Ixy [slug·ft²]", inertia.Ixy_kgm2 / SLUGFT2_TO_KGM2 if inertia.Ixy_kgm2 is not None else None)   # Likely has symmetry, but allow user to enter if known
        Ixz = input_value("Ixz [slug·ft²]", inertia.Ixz_kgm2 / SLUGFT2_TO_KGM2 if inertia.Ixz_kgm2 is not None else None)
        Iyz = input_value("Iyz [slug·ft²]", inertia.Iyz_kgm2 / SLUGFT2_TO_KGM2 if inertia.Iyz_kgm2 is not None else None)

        if Ixx is not None:
            Ixx *= SLUGFT2_TO_KGM2
        if Iyy is not None:
            Iyy *= SLUGFT2_TO_KGM2
        if Izz is not None:
            Izz *= SLUGFT2_TO_KGM2
        if Ixy is not None:
            Ixy *= SLUGFT2_TO_KGM2
        if Ixz is not None:
            Ixz *= SLUGFT2_TO_KGM2
        if Iyz is not None:
            Iyz *= SLUGFT2_TO_KGM2

    return Inertia(
        Ixx_kgm2=Ixx,
        Iyy_kgm2=Iyy,
        Izz_kgm2=Izz,
        Ixy_kgm2=Ixy,
        Ixz_kgm2=Ixz,
        Iyz_kgm2=Iyz,
    )


def populate_dataclass(instance):
    """Populate any dataclass from user input."""
    for field in fields(instance):
        current_value = getattr(instance, field.name)
        value = input_value(f"{field.name}", current_value)
        setattr(instance, field.name, value)
    return instance

def edit_longitudinal(
    units: str,
    longitudinal: LongitudinalCoefficients | None = None
    ) -> LongitudinalCoefficients:
    """Modify the longitudinal coefficients of an aircraft."""
    print_section("Longitudinal Coefficients")

    if longitudinal is None:
        longitudinal = LongitudinalCoefficients()

    longitudinal.CX0 = input_value("CX0", longitudinal.CX0)
    longitudinal.CZ0 = input_value("CZ0", longitudinal.CZ0)
    longitudinal.Cm0 = input_value("Cm0", longitudinal.Cm0)

    for name in ("CX", "CZ", "Cm"):
        print(f"\n{name} Derivatives")
        populate_dataclass(getattr(longitudinal, name))

    return longitudinal

def edit_lateral(
    units: str,
    lateral: LateralCoefficients | None = None
    ) -> LateralCoefficients:
    """Modify the lateral coefficients of an aircraft."""
    print_section("Lateral Coefficients")

    if lateral is None:
        lateral = LateralCoefficients()

    lateral.CY0 = input_value("CY0", lateral.CY0)
    lateral.Cl0 = input_value("Cl0", lateral.Cl0)
    lateral.Cn0 = input_value("Cn0", lateral.Cn0)

    for name in ("CY", "Cl", "Cn"):
        print(f"\n{name} Derivatives")
        populate_dataclass(getattr(lateral, name))

    return lateral

# =============================================================================
# Menu Actions
# =============================================================================

def list_aircraft():
    """Display the aircraft library."""
    print_page("Aircraft Library")
    files = print_aircraft_list()
    if not files:
        return


def select_current_aircraft():
    """Prompt the user to choose the current aircraft."""
    aircraft = choose_aircraft("Select Current Aircraft")
    if aircraft is None:
        return
    select_aircraft(aircraft)


def view_aircraft(units: str):
    """Display the details of a chosen aircraft."""
    aircraft = choose_aircraft("View Aircraft")
    if aircraft is None:
        return
    print_aircraft(aircraft, units)


def create_aircraft(units: str):
    """Interactively create a new aircraft."""
    print_page("Create New Aircraft")
    print(f"Preferred units: {units}")
    print()

    metadata = edit_metadata()
    print()
    geometry = edit_geometry(units)
    print()
    inertia = edit_inertia(units)
    print()

    aerodynamics = AeroCoefficients()
    if ask_yes_no("Would you like to enter known aerodynamic coefficients?"):
        print()
        aerodynamics.longitudinal = edit_longitudinal(units)
        print()
        aerodynamics.lateral = edit_lateral(units)
    print()

    ts = timestamp()

    aircraft = Aircraft(
        metadata=Metadata(
            name=metadata.name,
            description=metadata.description,
            created=ts,
            modified=ts
        ),
        geometry=geometry,
        inertia=inertia,
        aerodynamics=aerodynamics
    )

    if ask_yes_no("Save aircraft?", default=True):
        save_aircraft(aircraft, overwrite=False)
    else:
        print("\nAircraft creation cancelled.")


def edit_aircraft_dialog(aircraft: Aircraft, units: str):
    original_name = aircraft.metadata.name
    modified = False

    while True:
        print_page(f"Edit Aircraft: {aircraft.metadata.name}")
        print(" 1. Save")
        print(" 2. Metadata")
        print(" 3. Geometry")
        print(" 4. Inertia")
        print(" 5. Longitudinal Coefficients")
        print(" 6. Lateral Coefficients")
        print(" 0. Cancel")

        choice = input("\nSelection: ")
        print()

        if choice == "1":
            if not modified:
                print("No changes to save.")
                pause()
                continue

            if save_aircraft(aircraft, overwrite=True, original_name=original_name):
                original_name = aircraft.metadata.name
                modified = False
                print("Aircraft changes saved.")
                return

        elif choice == "2":
            aircraft.metadata = edit_metadata(aircraft.metadata)
            modified = True

        elif choice == "3":
            aircraft.geometry = edit_geometry(units, aircraft.geometry)
            modified = True
    
        elif choice == "4":
            aircraft.inertia = edit_inertia(units, aircraft.inertia)
            modified = True

        elif choice == "5":
            aircraft.aerodynamics.longitudinal = edit_longitudinal(units, aircraft.aerodynamics.longitudinal)
            modified = True

        elif choice == "6":
            aircraft.aerodynamics.lateral = edit_lateral(units, aircraft.aerodynamics.lateral)
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

def edit_aircraft(units: str):
    """Edit an existing aircraft."""
    aircraft = choose_aircraft("Edit Aircraft")
    if aircraft is None:
        return
    edit_aircraft_dialog(aircraft, units)


def delete_aircraft():
    """Delete an aircraft from the aircraft library."""
    aircraft = choose_aircraft("Delete Aircraft")
    if aircraft is None:
        return

    name = aircraft.metadata.name

    # Prevent deleting the active aircraft
    if get_current_aircraft_name() == name:
        print("\nCannot delete the current aircraft.")
        print("First select a different aircraft as the current one.")
        return

    if not ask_yes_no(f"Are you sure you want to permanently delete '{name}'?"):
        print("\nDeletion cancelled.")
        return

    aircraft_file = AIRCRAFT_LIBRARY / f"{name}.json"
    try:
        aircraft_file.unlink()
        print(f"\n'{name}' deleted successfully.")
    except OSError as e:
        print("\nFailed to delete aircraft.")
        print(e)

# =============================================================================
# Menus
# =============================================================================

def clear_screen():
    """Clear the terminal screen."""
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        # Fall back to printing blank lines
        print("\n" * 100)


def print_header():
    print("=" * SCREEN_WIDTH)
    print("AIRCRAFT LIBRARY MANAGER")
    print("=" * SCREEN_WIDTH)


def print_developer_info():
    print("Developed by Xander D. Mosley")
    print("-" * SCREEN_WIDTH)
    print()


def print_menu():
    print(" 1. List aircraft")
    print(" 2. Select current aircraft")
    print(" 3. View aircraft")
    print(" 4. Create aircraft")
    print(" 5. Edit aircraft")
    print(" 6. Delete aircraft")
    print(" 0. Quit")


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


def main():
    AIRCRAFT_LIBRARY.mkdir(exist_ok=True)

    settings = load_user_settings()
    units = settings.get("preferred_units", "metric").lower()

    while True:
        clear_screen()
        print_header()
        print_developer_info()
        print_menu()

        choice = input("\nSelection: ").strip()

        if choice == "1":
            list_aircraft()

        elif choice == "2":
            select_current_aircraft()

        elif choice == "3":
            view_aircraft(units)

        elif choice == "4":
            create_aircraft(units)

        elif choice == "5":
            edit_aircraft(units)

        elif choice == "6":
            delete_aircraft()

        elif choice == "0":
            break

        else:
            print("Invalid selection.")

        pause()

# =============================================================================

if __name__ == "__main__":
    main()