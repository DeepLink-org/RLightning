import pytest

from rlightning.utils.placement.resource_pool import NodeResource, ResourcePool

# =============================================================================
# NodeResource Tests
# =============================================================================


class TestNodeResource:
    """Tests for NodeResource."""

    def test_node_resource_creation(self):
        """Test basic NodeResource creation."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
        )
        assert node.node_id == "node_0"
        assert node.total_gpus == 8
        assert node.available_gpus == 8
        assert not node.is_empty

    def test_has_resources(self):
        """Test resource availability check."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
        )
        assert node.has_resources(gpus=4)
        assert node.has_resources(gpus=8)
        assert not node.has_resources(gpus=9)

    def test_allocate_basic(self):
        """Test basic GPU allocation - now modifies self directly."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
        )
        node.allocate(gpus=4, component_types=["train"])

        assert node.gpu_cursor == 4  # Cursor advanced
        assert node.available_gpus == 4  # Remaining GPUs
        assert "train" in node.allocations
        assert node.allocations["train"] == [(0, 3)]

    def test_allocate_overlapping_with_consume_false(self):
        """Test overlapping allocation using consume=False for colocate mode."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
        )
        # Allocate eval without consuming
        node.allocate(gpus=4, component_types=["eval"], consume=False)
        # Allocate env on the same GPUs without consuming
        node.allocate(gpus=4, component_types=["env"], consume=False)

        assert "eval" in node.allocations
        assert "env" in node.allocations
        # Both should have the same range (overlapping)
        assert node.allocations["eval"] == [(0, 3)]
        assert node.allocations["env"] == [(0, 3)]
        # Cursor not advanced since consume=False
        assert node.gpu_cursor == 0
        assert node.available_gpus == 8

    def test_allocate_separate_components(self):
        """Test separate allocation for multiple components."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
        )
        node.allocate(gpus=[2, 2], component_types=["eval", "env"])

        assert node.allocations["eval"] == [(0, 1)]
        assert node.allocations["env"] == [(2, 3)]
        assert node.available_gpus == 4

    def test_allocate_exceeds_available(self):
        """Test allocation exceeding available resources raises error."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
            gpu_cursor=4,  # Already allocated 4, only 4 remaining
        )
        with pytest.raises(RuntimeError):
            node.allocate(gpus=5, component_types=["train"])

    def test_allocate_zero_gpus(self):
        """Test allocation of 0 GPUs logs warning and does nothing."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
        )
        node.allocate(gpus=0)
        # Cursor should not advance
        assert node.gpu_cursor == 0

    def test_is_empty(self):
        """Test is_empty property."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
            gpu_cursor=8,  # All GPUs allocated
        )
        assert node.is_empty

    def test_copy(self):
        """Test deep copy of NodeResource."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
            allocations={"train": [(0, 3)]},
            gpu_cursor=4,
        )
        copied = node.copy()

        assert copied.node_id == node.node_id
        assert copied.allocations == node.allocations
        assert copied.allocations is not node.allocations  # Deep copy
        assert copied.gpu_cursor == node.gpu_cursor

    def test_component_types_property(self):
        """Test component_types property returns keys from allocations."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
            allocations={"train": [(0, 3)], "buffer": [(4, 4)]},
        )
        assert set(node.component_types) == {"train", "buffer"}

    def test_max_allocated_gpus(self):
        """Test max_allocated_gpus tracks the maximum allocation."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
        )
        # First allocation with consume=False
        node.allocate(gpus=4, component_types=["train"], consume=False)
        assert node.max_allocated_gpus == 4
        assert node.gpu_cursor == 0  # Not consumed

        # Second allocation with consume=False (overlapping)
        node.allocate(gpus=6, component_types=["eval"], consume=False)
        assert node.max_allocated_gpus == 6  # Updated to max
        assert node.gpu_cursor == 0  # Still not consumed

    def test_allocate_consume_true(self):
        """Test allocation with consume=True advances cursor."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
        )
        node.allocate(gpus=4, component_types=["train"], consume=True)
        assert node.gpu_cursor == 4
        assert node.available_gpus == 4
        assert node.max_allocated_gpus == 4


# =============================================================================
# ResourcePool Tests
# =============================================================================


class TestResourcePool:
    """Tests for ResourcePool."""

    def test_resource_pool_creation(self):
        """Test basic ResourcePool creation with auto-inferred component_types."""
        nodes = [
            NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
                allocations={"train": [(0, 3)]},
                gpu_cursor=4,  # 4 GPUs allocated
            )
        ]
        pool = ResourcePool(name="test_pool", nodes=nodes)

        assert pool.name == "test_pool"
        assert pool.num_nodes == 1
        assert pool.total_gpus == 8
        assert "train" in pool.component_types
        assert "buffer" in pool.component_types  # Auto-added when train exists

    def test_get_component_indices(self):
        """Test getting component GPU indices."""
        nodes = [
            NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
                allocations={"train": [(0, 3)]},
                gpu_cursor=4,
            )
        ]
        pool = ResourcePool(name="test_pool", nodes=nodes)

        indices = pool.get_component_indices("train")
        assert indices == "0-3"

    def test_get_component_indices_invalid_component(self):
        """Test getting indices for non-existent component."""
        nodes = [
            NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
                allocations={"train": [(0, 3)]},
                gpu_cursor=4,
            )
        ]
        pool = ResourcePool(name="test_pool", nodes=nodes)

        indices = pool.get_component_indices("eval")
        assert indices == ""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        nodes = [
            NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
                allocations={"train": [(0, 3)]},
                gpu_cursor=4,
            )
        ]
        pool = ResourcePool(name="test_pool", nodes=nodes)

        d = pool.to_dict()
        assert d["name"] == "test_pool"
        assert d["num_node"] == 1
        assert d["num_gpus"] == 8
        assert "train" in d

    def test_to_yaml_dict(self):
        """Test YAML-friendly serialization."""
        nodes = [
            NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
                allocations={"train": [(0, 3)]},
                gpu_cursor=4,
            )
        ]
        pool = ResourcePool(name="test_pool", nodes=nodes)

        yaml_d = pool.to_yaml_dict()
        assert yaml_d["name"] == "test_pool"
        assert yaml_d["num_node"] == 1
        assert yaml_d["num_gpus"] == 8
        assert "train" in yaml_d

    def test_yaml_dict_heterogeneous_nodes(self):
        """Test to_yaml_dict with heterogeneous node GPU counts."""
        nodes = [
            NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
                allocations={"train": [(0, 7)]},
                gpu_cursor=8,
            ),
            NodeResource(
                node_id="node_1",
                ip="192.168.1.2",
                total_cpus=64,
                total_gpus=8,
                allocations={"train": [(0, 3)]},
                gpu_cursor=4,
            ),
        ]
        pool = ResourcePool(name="train_pool", nodes=nodes)

        yaml_dict = pool.to_yaml_dict()
        assert yaml_dict["name"] == "train_pool"
        assert yaml_dict["num_node"] == 2
        assert yaml_dict["num_gpus"] == 8
        assert yaml_dict["train"] == "0-7, 8-11"


# =============================================================================
# ResourcePool YAML Parsing Tests
# =============================================================================


class TestResourcePoolYamlParsing:
    """Tests for ResourcePool YAML parsing methods."""

    def test_parse_index_str_single_range(self):
        """Test parsing single range string."""
        result = ResourcePool._parse_index_str("0-7")
        assert result == [(0, 7)]

    def test_parse_index_str_multiple_ranges(self):
        """Test parsing multiple ranges string."""
        result = ResourcePool._parse_index_str("0-3, 8-11")
        assert result == [(0, 3), (8, 11)]

    def test_parse_index_str_single_number(self):
        """Test parsing single number."""
        result = ResourcePool._parse_index_str("5")
        assert result == [(5, 5)]

    def test_parse_index_str_integer_input(self):
        """Test parsing integer input."""
        result = ResourcePool._parse_index_str(5)
        assert result == [(5, 5)]

    def test_parse_index_str_none(self):
        """Test parsing None input."""
        result = ResourcePool._parse_index_str(None)
        assert result == []

    def test_parse_index_str_empty(self):
        """Test parsing empty string."""
        result = ResourcePool._parse_index_str("")
        assert result == []

    def test_parse_index_str_reversed_range(self):
        """Test parsing reversed range (end < start)."""
        result = ResourcePool._parse_index_str("7-0")
        assert result == [(0, 7)]  # Should be normalized

    def test_split_global_range_single_node(self):
        """Test splitting global range for single node."""
        # Single node with 8 GPUs: offsets = [0, 8]
        result = ResourcePool._split_global_range_by_nodes(0, 7, [0, 8])
        assert result == [(0, 0, 7)]  # (node_idx, local_start, local_end)

    def test_split_global_range_multi_node(self):
        """Test splitting global range across multiple nodes."""
        # Two nodes with 4 GPUs each: offsets = [0, 4, 8]
        result = ResourcePool._split_global_range_by_nodes(2, 6, [0, 4, 8])
        # Should split: node 0 gets [2, 3], node 1 gets [0, 2]
        assert result == [(0, 2, 3), (1, 0, 2)]

    def test_split_global_range_exact_node_boundary(self):
        """Test splitting range that ends exactly at node boundary."""
        # Two nodes with 4 GPUs each: offsets = [0, 4, 8]
        result = ResourcePool._split_global_range_by_nodes(0, 3, [0, 4, 8])
        assert result == [(0, 0, 3)]  # Only first node

    def test_from_yaml_dict_basic(self):
        """Test creating ResourcePool from YAML dict."""
        cluster_nodes = {
            "node_0": NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
            ),
            "node_1": NodeResource(
                node_id="node_1",
                ip="192.168.1.2",
                total_cpus=64,
                total_gpus=8,
            ),
        }

        pool_cfg = {
            "name": "train_pool",
            "num_node": 1,
            "num_gpus": 8,
            "train": "0-7",
        }

        pool = ResourcePool.from_yaml_dict(pool_cfg, cluster_nodes)
        assert pool.name == "train_pool"
        assert pool.num_nodes == 1
        assert "train" in pool.component_types
        assert "buffer" in pool.component_types  # Auto-added when train exists

    def test_from_yaml_dict_with_node_ids(self):
        """Test creating ResourcePool with explicit node_ids."""
        cluster_nodes = {
            "node_0": NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
            ),
            "node_1": NodeResource(
                node_id="node_1",
                ip="192.168.1.2",
                total_cpus=64,
                total_gpus=4,
            ),
        }

        pool_cfg = {
            "name": "train_pool",
            "num_node": 2,
            "num_gpus": [8, 4],
            "node_ids": ["node_0", "node_1"],
            "train": "0-11",
        }

        pool = ResourcePool.from_yaml_dict(pool_cfg, cluster_nodes)
        assert pool.name == "train_pool"
        assert pool.num_nodes == 2
        assert pool.node_ids == ["node_0", "node_1"]

    def test_from_yaml_dict_multi_component(self):
        """Test creating ResourcePool with multiple components."""
        cluster_nodes = {
            "node_0": NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
            ),
        }

        pool_cfg = {
            "name": "rollout_pool",
            "num_node": 1,
            "num_gpus": 8,
            "eval": "0-3",
            "env": "4-7",
        }

        pool = ResourcePool.from_yaml_dict(pool_cfg, cluster_nodes)
        assert pool.name == "rollout_pool"
        assert "eval" in pool.component_types
        assert "env" in pool.component_types

    def test_from_yaml_dict_not_enough_nodes(self):
        """Test error when not enough nodes available."""
        cluster_nodes = {
            "node_0": NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
            ),
        }

        pool_cfg = {
            "name": "train_pool",
            "num_node": 2,  # Requires 2 nodes but only 1 available
            "num_gpus": 8,
            "train": "0-7",
        }

        with pytest.raises(ValueError, match="Not enough nodes"):
            ResourcePool.from_yaml_dict(pool_cfg, cluster_nodes)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_node_resource_multiple_allocations(self):
        """Test multiple sequential allocations on a node - now modifies self directly."""
        node = NodeResource(
            node_id="node_0",
            ip="192.168.1.1",
            total_cpus=64,
            total_gpus=8,
        )

        # First allocation - modifies node directly
        node.allocate(gpus=4, component_types=["train"])
        assert node.available_gpus == 4
        # Allocations recorded on the node
        assert node.allocations["train"] == [(0, 3)]

        # Second allocation - continues from cursor position
        node.allocate(gpus=2, component_types=["eval"])
        assert node.available_gpus == 2

        # Second allocation uses cursor position (continues from 4)
        assert node.allocations["eval"] == [(4, 5)]

        # Cursor tracks total allocated
        assert node.gpu_cursor == 6  # 4 + 2

    def test_resource_pool_multi_node(self):
        """Test resource pool with multiple nodes."""
        nodes = [
            NodeResource(
                node_id="node_0",
                ip="192.168.1.1",
                total_cpus=64,
                total_gpus=8,
                allocations={"train": [(0, 3)]},
                gpu_cursor=4,
            ),
            NodeResource(
                node_id="node_1",
                ip="192.168.1.2",
                total_cpus=64,
                total_gpus=8,
                allocations={"train": [(0, 3)]},
                gpu_cursor=4,
            ),
        ]
        pool = ResourcePool(name="test_pool", nodes=nodes)

        assert pool.num_nodes == 2
        assert pool.total_gpus == 16
        indices = pool.get_component_indices("train")
        # Should have indices from both nodes in pool-global space
        assert "0-3" in indices
        assert "8-11" in indices
