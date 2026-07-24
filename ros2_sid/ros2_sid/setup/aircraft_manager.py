#!/usr/bin/env python3

"""
aircraft_manager.py — Manage the aircraft library.

Features
--------
- Create aircraft definitions
- Edit aircraft definitions
- View aircraft information
- Delete aircraft
- Select current aircraft
- Store all aircraft as JSON files
- Convert all user inputs to SI units

Author
------
Xander D. Mosley  
Email: XanderDMosley.Engineer@gmail.com  
Date: 22 Jul 2026
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
import json
import shutil

# __all__ = []
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

AIRCRAFT_LIBRARY = SCRIPT_DIR / "aircraft_library"
CURRENT_AIRCRAFT = SCRIPT_DIR / "current_aircraft.json"

AIRCRAFT_LIBRARY.mkdir(exist_ok=True)

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
    manufacturer: str = ""
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
    Ixz_kgm2: float | None = None

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
    footnotes: str = ""


# =============================================================================
# Utility Functions
# =============================================================================

def clear_screen():
    """Clear the terminal."""
    pass


def pause():
    """Wait for the user before continuing."""
    input("\nPress Enter to continue...")


def timestamp():
    """Return ISO timestamp."""
    return datetime.now().isoformat(timespec="seconds")


# =============================================================================
# File Functions
# =============================================================================

def save_aircraft(aircraft: Aircraft):
    """Save an aircraft to the aircraft library."""
    pass


def load_aircraft(name: str) -> Aircraft:
    """Load an aircraft from disk."""
    pass


def list_aircraft():
    """Return a list of saved aircraft."""
    pass


def select_aircraft(name: str):
    """Copy selected aircraft to current_aircraft.json."""
    pass


# =============================================================================
# Input Functions
# =============================================================================

def choose_units():
    """
    Ask user for Metric or Imperial units.

    Returns
    -------
    str
        "metric" or "imperial"
    """
    pass


def input_geometry(units):
    """Collect aircraft geometry."""
    pass


def input_inertia(units):
    """Collect inertia tensor."""
    pass


def input_longitudinal():
    """Collect known longitudinal coefficients."""
    pass


def input_lateral():
    """Collect known lateral coefficients."""
    pass


# =============================================================================
# Menu Actions
# =============================================================================

def create_aircraft():
    """Interactive aircraft creation."""
    pass


def edit_aircraft():
    """Edit an existing aircraft."""
    pass


def view_aircraft():
    """Display aircraft information."""
    pass


def delete_aircraft():
    """Delete an aircraft."""
    pass


def select_current_aircraft():
    """Choose current aircraft."""
    pass


# =============================================================================
# Menus
# =============================================================================

def print_header():

    print("=" * 60)
    print("Aircraft Library Manager")
    print("=" * 60)


def print_menu():

    print()
    print("1. List aircraft")
    print("2. Create aircraft")
    print("3. View aircraft")
    print("4. Edit aircraft")
    print("5. Delete aircraft")
    print("6. Select current aircraft")
    print("0. Quit")
    print()


def main():

    while True:

        clear_screen()

        print_header()

        print_menu()

        choice = input("Selection: ").strip()

        if choice == "1":
            list_aircraft()
            pause()

        elif choice == "2":
            create_aircraft()

        elif choice == "3":
            view_aircraft()
            pause()

        elif choice == "4":
            edit_aircraft()

        elif choice == "5":
            delete_aircraft()

        elif choice == "6":
            select_current_aircraft()

        elif choice == "0":
            break

        else:
            print("Invalid selection.")
            pause()


# =============================================================================

if __name__ == "__main__":
    main()