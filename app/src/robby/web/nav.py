"""Navigation per role (ui-mockups extension §1), landing pages and the header identity lines."""

from __future__ import annotations

from dataclasses import dataclass

from robby.models import AppUser, Role, role_label


@dataclass(frozen=True)
class NavItem:
    label: str
    path: str


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
    Role.CLINICIAN: (
        NavItem("New admission", "/admissions/new"),
        NavItem("Recent admissions", "/admissions"),
        NavItem("Import from Excel", "/admissions/import"),
        NavItem("Readiness board", "/readiness"),
    ),
    Role.FACILITY_ADMIN: (
        NavItem("Audit report", "/audit"),
        NavItem("Unclassified records", "/audit/unclassified"),
        NavItem("Readiness board", "/readiness"),
        NavItem("Users", "/admin/users"),
        NavItem("Integrations", "/admin/integrations"),
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
    """Where a role starts (spec §5): the first item of its navigation."""
    return NAV[role][0].path


def _matches(path: str, item_path: str) -> bool:
    return path == item_path or path.startswith(item_path.rstrip("/") + "/")


def nav_for(role: Role, current_path: str) -> list[NavLink]:
    items = NAV[role]
    matching = [item for item in items if _matches(current_path, item.path)]
    active = max(matching, key=lambda item: len(item.path)) if matching else None
    return [NavLink(item.label, item.path, item is active) for item in items]


def who_for(user: AppUser) -> Who:
    if user.facility is not None:
        place = user.facility.name
    elif user.role is Role.NATIONAL_REVIEWER:
        place = "All facilities · National"
    else:
        place = "All facilities"
    return Who(place, role_label(user.role))
