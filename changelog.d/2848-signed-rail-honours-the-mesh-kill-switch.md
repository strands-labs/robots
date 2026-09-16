### Fixed: the dashboard's signed safety rail honours the mesh kill switch

`STRANDS_MESH=false` is documented as a hard kill switch, and
`mesh_disabled_by_env()` is the one predicate every path which can open a
session asks - #2515 added the check after `robot_mesh`'s gateway joined the
fleet without it. `MeshBridge.start()` asks it. `MeshBridge._safety_mesh()`,
the dashboard's second `Mesh` construction site, did not:

```python
os.environ["STRANDS_MESH"] = "false"
MeshBridge(peer_id="dash").signed_estop()
# before: a live "dash-safety" gateway peer joins the fleet
# after:  {"signed": False, "error": "signed safety rail disabled by STRANDS_MESH=false"}
```

So an operator who asked for no mesh still got one, created by the first
e-stop - the ghost-peer case the switch was added to close, arriving on the path
where the peer list is least likely to be watched and by way of the action least
likely to be debugged afterwards. The switch is now read before the constructor,
because constructing a `Mesh` is what joins.

The refusal names the variable rather than reporting "safety mesh unavailable",
since an operator who set the switch would otherwise go hunting a fault instead
of looking at the switch they set; a rail that is genuinely broken still reports
"unavailable", so the two answers stay distinguishable. The per-peer broadcast
stop is unaffected and still fans out.

The new site is graded from the source rather than only by behaviour, so a third
construction site added without the predicate fails on arrival - a missing case
for an untested site is how this one stayed ungated while its neighbour was
gated.
