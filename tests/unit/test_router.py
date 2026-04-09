from rlightning.policy.utils.router import NodeAffinityRouter, SimpleRouter


def test_simple_router_balances_assignments_against_current_loads():
    router = SimpleRouter()
    current_loads = [2, 0, 1]

    assignments = router.assign(current_loads, num_tasks=4)

    final_loads = [2, 0, 1]
    for idx in assignments:
        final_loads[idx] += 1

    assert assignments[0] == 1
    assert max(final_loads) - min(final_loads) <= 1


def test_node_affinity_router_prefers_policies_on_the_same_node():
    router = NodeAffinityRouter(
        component_distribution={
            "node-a": {"env": {"ids": [0, 1]}},
            "node-b": {"env": {"ids": [2]}},
        },
        policy_node_ids=["node-a", "node-b", "node-a"],
    )
    current_loads = [0, 0, 1]

    assignments = router.assign(current_loads, num_tasks=3, env_ids=["env-0", "env-2", "env-1"])

    assert assignments == [0, 1, 0]
    assert current_loads == [2, 1, 1]


def test_node_affinity_router_falls_back_to_simple_routing_without_env_ids():
    router = NodeAffinityRouter(component_distribution={}, policy_node_ids=["node-a", "node-b"])
    current_loads = [1, 0]

    assignments = router.assign(current_loads, num_tasks=2, env_ids=None)

    assert assignments == [1, 0]
