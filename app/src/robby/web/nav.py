"""Navigation per role (each role sees only its own pages), landing pages and the header identity
lines."""

from __future__ import annotations

from dataclasses import dataclass

from robby.models import AppUser, Role, role_label


@dataclass(frozen=True)
class NavItem:
    label: str
    path: str
    also: tuple[str, ...] = ()  # other path prefixes that belong to this item


@dataclass(frozen=True)
class NavLink:
    label: str
    path: str
    active: bool


@dataclass(frozen=True)
class Who:
    place: str
    role: str


NAV: dict[Role, tuple[NavItem, ...]] = {
    Role.LAB_TECHNICIAN: (
        NavItem("New admission", "/admissions/new"),
        NavItem("Recent entries", "/admissions"),
    ),
    Role.CLINICIAN: (NavItem("Find patient", "/patients", also=("/admissions",)),),
    Role.FACILITY_ADMIN: (
        NavItem("Hospital report", "/audit"),
        NavItem("Users", "/admin/users"),
    ),
    Role.MODEL_CUSTODIAN: (
        NavItem("Models", "/models"),
        NavItem("Monitoring", "/monitoring"),
        NavItem("Approval log", "/approval-log"),
    ),
    Role.NATIONAL_REVIEWER: (NavItem("National overview", "/national"),),
    Role.SYSTEM_ADMIN: (
        NavItem("Facilities", "/admin/facilities"),
        NavItem("Users", "/admin/users"),
    ),
}


def landing_path(role: Role) -> str:
    """Where a role starts after sign-in: the first item of its navigation."""
    return NAV[role][0].path


def _matches(path: str, item_path: str) -> bool:
    return path == item_path or path.startswith(item_path.rstrip("/") + "/")


def _match_length(path: str, item: NavItem) -> int:
    """Length of the longest of the item's paths that `path` falls under, 0 for none."""
    return max((len(p) for p in (item.path, *item.also) if _matches(path, p)), default=0)


def nav_for(role: Role, current_path: str) -> list[NavLink]:
    items = NAV[role]
    lengths = {item.path: _match_length(current_path, item) for item in items}
    matching = [item for item in items if lengths[item.path]]
    active = max(matching, key=lambda item: lengths[item.path]) if matching else None
    return [NavLink(item.label, item.path, item is active) for item in items]


def who_for(user: AppUser) -> Who:
    if user.facility is not None:
        place = user.facility.name
    elif user.role is Role.NATIONAL_REVIEWER:
        place = "All facilities · National"
    else:
        place = "All facilities"
    return Who(place, role_label(user.role))
