"""Tests for the MuJoCo simulation engine, asset manager, and zenoh mesh."""


class TestAssetManager:
    """Test the robot asset manager."""

    def test_get_assets_dir(self):
        from strands_robots.assets import get_assets_dir

        assets_dir = get_assets_dir()
        assert assets_dir.exists()
        assert assets_dir.is_dir()

    def test_list_available_robots(self):
        from strands_robots.assets import list_available_robots

        robots = list_available_robots()
        assert len(robots) >= 25
        names = [r["name"] for r in robots]
        assert "so100" in names
        assert "panda" in names

    def test_resolve_model_path(self):
        from strands_robots.assets import resolve_model_path

        path = resolve_model_path("so100")
        assert path is not None
        assert path.exists()
        assert path.suffix == ".xml"

    def test_resolve_robot_name_alias(self):
        from strands_robots.assets import resolve_robot_name

        # trs_so_arm100 → so100
        resolved = resolve_robot_name("trs_so_arm100")
        assert resolved == "so100"

    def test_resolve_robot_name_identity(self):
        from strands_robots.assets import resolve_robot_name

        resolved = resolve_robot_name("so100")
        assert resolved == "so100"

    def test_resolve_unknown_robot(self):
        from strands_robots.assets import resolve_model_path

        path = resolve_model_path("nonexistent_robot_xyz_123")
        assert path is None

    def test_get_search_paths(self):
        from strands_robots.assets import get_search_paths

        paths = get_search_paths()
        assert len(paths) >= 1
        assert paths[0].exists()

    def test_list_robots_by_category(self):
        from strands_robots.assets import list_robots_by_category

        categories = list_robots_by_category()
        assert isinstance(categories, dict)
        assert "arm" in categories
        assert len(categories["arm"]) >= 5

    def test_list_aliases(self):
        from strands_robots.assets import list_aliases

        aliases = list_aliases()
        assert "trs_so_arm100" in aliases  # dir alias → canonical
        assert "franka" in aliases or "franka_emika_panda" in aliases
        assert "g1" in aliases

    def test_get_robot_info(self):
        from strands_robots.assets import get_robot_info

        info = get_robot_info("so100")
        assert info is not None
        assert info["canonical_name"] == "so100"
        assert info["available"] is True
        assert info["category"] == "arm"

    def test_format_robot_table(self):
        from strands_robots.assets import format_robot_table

        table = format_robot_table()
        assert isinstance(table, str)
        assert "so100" in table
        assert "panda" in table


class TestSimulationImport:
    """Test that simulation module imports correctly."""

    def test_import_simulation_classes(self):
        from strands_robots.simulation import SimCamera, SimObject, SimRobot, SimStatus

        assert SimRobot is not None
        assert SimObject is not None
        assert SimCamera is not None
        assert SimStatus is not None

    def test_simulation_convenience_alias(self):
        """The module exports a `simulation` callable as convenience."""
        from strands_robots import simulation

        assert simulation is not None


class TestZenohMesh:
    """Test zenoh mesh networking module."""

    def test_import_zenoh_mesh(self):
        from strands_robots.zenoh_mesh import Mesh, init_mesh

        assert init_mesh is not None
        assert Mesh is not None

    def test_peer_info(self):
        from strands_robots.zenoh_mesh import PeerInfo

        peer = PeerInfo(peer_id="test-1", peer_type="robot", hostname="localhost")
        assert peer.peer_id == "test-1"
        assert peer.peer_type == "robot"

    def test_get_peers_empty(self):
        from strands_robots.zenoh_mesh import get_peers

        peers = get_peers()
        assert isinstance(peers, list)
