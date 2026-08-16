"""A model that cannot be resolved must say which of two causes it hit.

``MuJoCoSimEngine._unknown_model_msg`` serves two conditions with different
remedies:

* the registry does not know the name (a typo) - the remedy is a spelling fix,
  and naming close registry keys is what lets the caller make it in place;
* the registry knows the name and its model XML is not on disk - the name is
  already correct, so the remedy is the asset, not the spelling.

Only the second condition can put the requested name inside the candidate set
the suggestion is drawn from, and :func:`difflib.get_close_matches` scores an
exact match 1.0 and ranks it first. So the one case whose name needs no fixing
was the one told "Did you mean: <the name it just refused>", while the actual
cause - a missing asset directory the registry can name exactly - went
unmentioned. These tests pin both halves: the suggestion never echoes the
requested name, and a registered robot's refusal names the asset it wanted.
"""

import difflib
import pathlib

import pytest

from strands_robots.registry import get_robot
from strands_robots.simulation.base import close_match_hint
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

# Concrete registry subjects, one per ``auto_download`` posture. Their posture
# is asserted as a premise below rather than assumed, so editing robots.json
# fails here with a readable reason instead of quietly making a case vacuous.
HAND_PLACED_ROBOT = "google_robot"  # asset.auto_download is False
HAND_PLACED_ALIAS = "oxe_google"
DOWNLOADABLE_ROBOT = "panda"


def _offered(hint: str) -> list[str]:
    """The suggested names in a ``" Did you mean: a, b?"`` fragment, in order."""
    if "Did you mean:" not in hint:
        return []
    return [part.strip() for part in hint.split("Did you mean:")[1].split("?")[0].split(",")]


@pytest.fixture
def no_assets_on_disk(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Point the asset search at an empty directory, so every model is absent.

    Uses the real resolver against a real (empty) directory rather than doubling
    the presence probe, so the message under test is built from the same
    filesystem answer a user with un-downloaded assets gets.
    """
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
    return tmp_path


def test_the_registry_postures_these_tests_assume_still_hold() -> None:
    """Non-vacuity: both subjects must still carry the posture each case tests."""
    hand_placed = (get_robot(HAND_PLACED_ROBOT) or {}).get("asset") or {}
    downloadable = (get_robot(DOWNLOADABLE_ROBOT) or {}).get("asset") or {}
    assert hand_placed.get("auto_download") is False, (
        f"{HAND_PLACED_ROBOT!r} is no longer an auto_download=false entry; pick another "
        "subject for the hand-placed case."
    )
    assert downloadable, f"{DOWNLOADABLE_ROBOT!r} has no asset entry"
    assert downloadable.get("auto_download") is not False, (
        f"{DOWNLOADABLE_ROBOT!r} became an auto_download=false entry; pick another subject for the downloadable case."
    )


class TestARegisteredRobotIsNotToldToFixItsName:
    """The refusal for a known robot with a missing asset diagnoses the asset."""

    def test_the_refusal_does_not_suggest_the_name_it_refused(self, no_assets_on_disk: pathlib.Path) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg(HAND_PLACED_ROBOT)
        suggested = msg.split("Did you mean:")[1].split("?")[0] if "Did you mean:" in msg else ""
        assert HAND_PLACED_ROBOT not in suggested, (
            f"the refusal offered {HAND_PLACED_ROBOT!r} as a correction for itself: {msg}"
        )

    def test_the_refusal_names_the_asset_file_it_looked_for(self, no_assets_on_disk: pathlib.Path) -> None:
        asset = (get_robot(HAND_PLACED_ROBOT) or {})["asset"]
        msg = MuJoCoSimEngine._unknown_model_msg(HAND_PLACED_ROBOT)
        assert f"{asset['dir']}/{asset['model_xml']}" in msg, msg
        assert str(no_assets_on_disk) in msg, msg

    def test_the_refusal_says_the_robot_is_registered(self, no_assets_on_disk: pathlib.Path) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg(HAND_PLACED_ROBOT)
        assert "registered" in msg, msg
        assert "No model found" not in msg, f"a robot the registry knows was reported as an unknown name: {msg}"

    def test_a_hand_placed_entry_is_told_to_place_the_file(self, no_assets_on_disk: pathlib.Path) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg(HAND_PLACED_ROBOT)
        assert "auto_download=false" in msg, msg
        assert "download_assets" not in msg, (
            f"an entry that never auto-downloads was pointed at the download tool: {msg}"
        )

    def test_a_downloadable_entry_is_pointed_at_the_download_tool(self, no_assets_on_disk: pathlib.Path) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg(DOWNLOADABLE_ROBOT)
        assert "download_assets" in msg, msg
        assert f"robots='{DOWNLOADABLE_ROBOT}'" in msg, msg

    def test_an_alias_reports_the_canonical_entrys_asset(self, no_assets_on_disk: pathlib.Path) -> None:
        asset = (get_robot(HAND_PLACED_ROBOT) or {})["asset"]
        msg = MuJoCoSimEngine._unknown_model_msg(HAND_PLACED_ALIAS)
        assert f"{asset['dir']}/{asset['model_xml']}" in msg, msg
        assert HAND_PLACED_ALIAS in msg, msg


class TestATypoStillGetsSpellingHelp:
    """Over-reach controls: the unknown-name condition is unchanged."""

    def test_a_near_miss_still_suggests_close_registry_keys(self) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg("so10x")
        assert "No model found for 'so10x'." in msg, msg
        assert "so101" in msg or "so100" in msg, msg
        assert "list_urdfs" in msg, msg

    def test_a_name_with_no_neighbours_omits_the_suggestion(self) -> None:
        msg = MuJoCoSimEngine._unknown_model_msg("zzqqxx0000nope")
        assert "Did you mean:" not in msg, msg
        assert "list_urdfs" in msg, msg

    def test_an_unreadable_registry_degrades_to_the_bare_form(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.registry as registry

        def _boom() -> list[dict[str, str]]:
            raise RuntimeError("registry exploded")

        monkeypatch.setattr(registry, "list_robots", _boom)
        msg = MuJoCoSimEngine._unknown_model_msg("anything")
        assert "No model found for 'anything'." in msg, msg
        assert "Did you mean:" not in msg, msg

    def test_a_non_string_name_is_reported_rather_than_raising(self) -> None:
        # ``registered`` is total, so a name that cannot be a registry key
        # reaches here; the asset probe must not gate on the name's type.
        assert "No model found" in MuJoCoSimEngine._unknown_model_msg(42)  # type: ignore[arg-type]
        assert "No model found" in MuJoCoSimEngine._unknown_model_msg(None)  # type: ignore[arg-type]

    def test_the_message_stays_ascii(self, no_assets_on_disk: pathlib.Path) -> None:
        assert MuJoCoSimEngine._unknown_model_msg(HAND_PLACED_ROBOT).isascii()
        assert MuJoCoSimEngine._unknown_model_msg("so10x").isascii()


class TestCloseMatchHintNeverEchoesTheRequestedName:
    """The shared suggestion helper owns the rule, so no caller has to know it."""

    def test_an_exact_match_is_not_offered_back(self) -> None:
        # Compared as whole names: "gripper_link" legitimately *contains*
        # "gripper", so a substring test would pass for the wrong reason.
        assert _offered(close_match_hint("gripper", ["gripper", "grip_pad", "gripper_link"])) == [
            "gripper_link",
            "grip_pad",
        ]

    def test_dropping_the_self_match_promotes_the_next_candidate(self) -> None:
        # Three suggestions are still rendered: the self-match is replaced, not
        # merely removed, so the caller loses none of the three slots.
        offered = _offered(close_match_hint("arm1", ["arm1", "arm2", "arm3", "arm4", "totally_other"]))
        assert "arm1" not in offered, offered
        assert len(offered) == 3, offered

    @pytest.mark.parametrize(
        "requested, known",
        [
            ("arm2", ["arm1", "arm3", "arm4"]),
            ("so10x", ["so100", "so101", "panda", "go1", "omx"]),
            ("wrist", ["wrist_cam", "wrist_link", "waist", "base"]),
        ],
    )
    def test_a_known_set_without_the_requested_name_is_unchanged(self, requested: str, known: list[str]) -> None:
        # The object / camera / robot callers pass what the world holds, which
        # never contains the requested name - their rendering must not shift.
        # Compared against the plain three-match call this helper made before
        # the self-match filter, rather than a hand-written expectation, so the
        # control cannot encode a difflib ordering of its own.
        unchanged = difflib.get_close_matches(requested, known, n=3, cutoff=0.4)
        assert _offered(close_match_hint(requested, known)) == unchanged

    def test_a_non_string_or_empty_known_set_still_yields_nothing(self) -> None:
        assert close_match_hint(42, ["arm1"]) == ""
        assert close_match_hint("arm1", []) == ""
