"""Tests for ``format_robot_table`` - column width handling (issue #113)."""

from __future__ import annotations

from strands_robots.registry.robots import (
    _FIXED_PREFIX_WIDTH,
    format_robot_table,
    list_robots,
)


class TestDefaultWidth:
    def test_default_max_line_length_is_bounded(self):
        table = format_robot_table()  # default max_width=100
        max_len = max(len(line) for line in table.split("\n"))
        # Allow a small margin - the rule is the longest line; data rows
        # should fit inside max_width + some padding for the header/rule.
        assert max_len <= 101, f"max line {max_len} exceeds 100 chars"

    def test_contains_header_and_total(self):
        table = format_robot_table()
        assert "Name" in table
        assert "Category" in table
        assert "Description" in table
        assert f"Total: {len(list_robots())} robots" in table

    def test_contains_all_categories(self):
        table = format_robot_table()
        # At least one of each category should be represented in the registry.
        for cat in ("arm", "humanoid", "hand"):
            assert cat in table


class TestNarrowWidth:
    def test_80_col_terminal_fits(self):
        table = format_robot_table(max_width=80)
        max_len = max(len(line) for line in table.split("\n"))
        # 80 is a hard target for narrow terminals; our rule is <= that + 1
        # (the ellipsis adds one wide char that may not be counted).
        assert max_len <= 81, f"max line {max_len} exceeds 80 chars"

    def test_descriptions_are_truncated_with_ellipsis(self):
        """Long descriptions should end with the truncation marker '...'."""
        narrow = format_robot_table(max_width=80)
        wide = format_robot_table(max_width=1000)
        # At least one row must have been truncated at narrow width.
        assert "..." in narrow
        # And that same row is longer in the wide rendering.
        assert "..." not in wide


class TestWideWidth:
    def test_wide_width_disables_truncation(self):
        table = format_robot_table(max_width=1000)
        assert "..." not in table

    def test_minimum_desc_width_is_enforced(self):
        """Even at absurdly narrow widths we keep a 20-char Description column
        rather than collapsing to zero."""
        table = format_robot_table(max_width=20)
        # Prefix alone is wider than 20; we clamp to
        # _FIXED_PREFIX_WIDTH + 20 so every row still shows some description.
        max_len = max(len(line) for line in table.split("\n"))
        assert max_len >= _FIXED_PREFIX_WIDTH + 20 - 1


class TestConsistency:
    def test_row_count_matches_registry(self):
        """The table should have (2 header + robots + 2 footer) lines.
        Categories with zero robots contribute no data rows."""
        table = format_robot_table()
        lines = table.split("\n")
        non_empty_rows = [line for line in lines[2:-2] if line.strip() and "Total:" not in line]
        assert len(non_empty_rows) == len(list_robots())


class TestAsciiOnlyAndAlignment:
    """The table is emitted to plain CLI/tool output, which the project
    requires to be ASCII-only (no emojis). Wide emoji markers also break
    monospace column alignment because ``str.ljust`` counts code points,
    not display cells.
    """

    def test_table_is_pure_ascii(self):
        """No emoji / box-drawing chars leak into CLI output."""
        table = format_robot_table(max_width=1000)
        assert table.isascii(), "format_robot_table emitted non-ASCII characters"

    def test_description_column_is_aligned_across_rows(self):
        """Every data row must start its Description column at the same
        offset. A code-point marker that renders two cells wide (e.g. an
        emoji) shifts later rows and breaks this invariant."""
        table = format_robot_table(max_width=1000)
        lines = table.split("\n")
        # Data rows live between the header+rule (first 2 lines) and the
        # blank+Total footer (last 2 lines).
        data_rows = [ln for ln in lines[2:-2] if ln.strip()]
        assert data_rows, "expected at least one robot row"
        # The Description text starts immediately after the fixed prefix.
        # Find each row's description by slicing at the prefix width and
        # asserting the prefix region contains no stray wide markers: the
        # character at the prefix boundary is part of the description, so
        # the rule line (all '-') and every prefix must be the same width.
        prefix_len = _FIXED_PREFIX_WIDTH
        for row in data_rows:
            # The Sim/Real columns only ever hold "yes" or spaces, so the
            # prefix must be exactly ASCII spaces/letters/digits up to the
            # description. Verify the prefix slice has no multi-cell glyphs
            # by requiring it to be ASCII (1 cell == 1 code point).
            assert row[:prefix_len].isascii()

    def test_sim_real_markers_use_ascii_token(self):
        """Robots with sim support are flagged with an ASCII token, not an
        emoji."""
        table = format_robot_table(max_width=1000)
        # so100 has sim support in the registry.
        so100_row = next(ln for ln in table.split("\n") if ln.startswith("so100 "))
        assert "yes" in so100_row


class TestCustomCategoryNotDropped:
    """A robot whose category is outside the common set must still get a row.

    ``register_robot`` accepts an arbitrary ``category`` string, so a user can
    register a robot under a category (e.g. ``"quadruped"``) that the table's
    preferred display order does not enumerate. Such a robot must still appear
    in the body - otherwise the visible rows under-count the footer ``Total``,
    producing a misleading discovery surface.
    """

    def _register(self, tmp_path, name, category):
        from strands_robots.registry import register_robot

        robot_dir = tmp_path / name
        robot_dir.mkdir(parents=True, exist_ok=True)
        (robot_dir / "bot.xml").write_text(f'<mujoco model="{name}"><worldbody/></mujoco>')
        register_robot(
            name=name,
            model_xml="bot.xml",
            asset_dir=str(robot_dir),
            category=category,
            joints=4,
            description=f"custom {category} robot",
        )

    def test_custom_category_robot_appears_in_body(self, tmp_path):
        self._register(tmp_path, "quadbot", "quadruped")
        table = format_robot_table(max_width=1000)
        assert "quadbot" in table, "robot with a custom category was dropped from the table body"

    def test_body_row_count_matches_total_with_custom_category(self, tmp_path):
        """The consistency invariant must survive custom categories: one body
        row per registered robot, matching the footer ``Total``."""
        self._register(tmp_path, "quadbot", "quadruped")
        table = format_robot_table(max_width=1000)
        lines = table.split("\n")
        non_empty_rows = [ln for ln in lines[2:-2] if ln.strip() and "Total:" not in ln]
        assert len(non_empty_rows) == len(list_robots())

    def test_missing_category_defaults_to_other_and_is_shown(self, tmp_path):
        """A robot registered with an empty category groups under ``"other"``
        (via ``list_robots_by_category``) and must still be rendered."""
        self._register(tmp_path, "mysterybot", "")
        table = format_robot_table(max_width=1000)
        assert "mysterybot" in table
