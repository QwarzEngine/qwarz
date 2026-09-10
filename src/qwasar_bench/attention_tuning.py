"""Process-local attention experiments; never modifies the installed donor files.

Patch configuration before constructing a generator. Existing CUDA graph slots
must not be reused across policies. Standalone and graph paths share grouping.
"""
import ast
from contextlib import contextmanager
import inspect
import textwrap


ALLOWED = {"block_n": (16, 32, 64, 128), "block_h": (1, 2, 4, 8, 16),
           "num_warps": (2, 4, 8), "num_stages": (1, 2, 3, 4),
           "num_splits": (8, 16, 24, 28, 32, 48, 64, 85, 96, 128)}


def validate_policy(policy):
    if not isinstance(policy, dict) or policy.keys() - {"default", "queries"}:
        raise ValueError("invalid attention policy")
    queries = policy.get("queries", {})
    if not isinstance(queries, dict) or any(str(q) not in {str(i) for i in range(1, 17)} for q in queries):
        raise ValueError("query overrides require lengths 1..16")
    for options in [policy.get("default", {}), *queries.values()]:
        if not isinstance(options, dict) or options.keys() - ALLOWED.keys():
            raise ValueError("unsupported attention option")
        if any(type(value) is not int or value not in ALLOWED[key] for key, value in options.items()):
            raise ValueError("unsupported attention value")


def select_options(policy, query_length):
    return policy.get("queries", {}).get(str(query_length), policy.get("default", {}))


def rewrite_attention(source, mode, capture=False):
    """Change only explicit launch configuration nodes; fail on donor drift."""
    if mode not in ("graph", "standalone"):
        raise ValueError("unsupported attention rewrite mode")
    tree = ast.parse(textwrap.dedent(source))
    counts = {"block_h": 0}
    if mode == "graph": counts.update(block_n=0, splits_cap=0, compile=0)

    def option(key, default):
        return ast.Call(func=ast.Attribute(value=ast.Call(func=ast.Name(id="_qwasar_options", ctx=ast.Load()),
            args=[ast.Name(id="q_len", ctx=ast.Load())], keywords=[]), attr="get", ctx=ast.Load()),
            args=[ast.Constant(key), default], keywords=[])

    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in counts and name != "compile":
                    counts[name] += 1
                    node.value = option("num_splits" if name == "splits_cap" else name, node.value)
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if (mode == "graph" and isinstance(node.func, ast.Name) and node.func.id == "_compile_kernel"
                and len(node.args) == 6 and isinstance(node.args[1], ast.Name)
                and node.args[1].id == "_paged_attn_decode_split_kernel"):
                counts["compile"] += 1
                node.args[-2] = option("num_warps", node.args[-2])
                node.args[-1] = option("num_stages", node.args[-1])
            return node

    tree = Rewrite().visit(tree)
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"expected one configuration site per option, found {counts}")
    if capture:
        if mode != "graph": raise ValueError("capture requires graph configuration")
        tree.body[0].body.append(ast.parse("_qwasar_capture_config(self, bsz, q_len, q)").body[0])
    return ast.unparse(ast.fix_missing_locations(tree))


def standalone_kernel(module, options):
    validate_policy({"default": options})
    namespace = dict(vars(module)) | {"_qwasar_options": lambda q: options}
    source = rewrite_attention(inspect.getsource(module.paged_attn_triton_decode), "standalone")
    exec(compile(source, "<qwasar-attention-standalone>", "exec"), namespace)
    return namespace["paged_attn_triton_decode"]


@contextmanager
def graph_attention_context(policy, *, capture=None):
    """Must enclose model lifetime, including warmup and all measured requests."""
    validate_policy(policy)
    if any("block_h" in options for options in [policy.get("default", {}), *policy.get("queries", {}).values()]):
        raise ValueError("head-group changes require matching the C++ native launch grid; standalone screening only")
    from exllamav3.modules.attention_fn import bc_attn, triton_paged
    original_configure = bc_attn.BCAttn._configure
    original_step = bc_attn.BCAttn.step
    original_decode = triton_paged.paged_attn_triton_decode
    configs = []

    def record_config(instance, bsz, q_len, q):
        instance._qwasar_queries = getattr(instance, "_qwasar_queries", {})
        if capture is not None: instance._qwasar_queries[(bsz, q_len)] = q
        configs.append({"query_length": q_len, "batch": bsz, "heads": instance.num_q_heads,
                        "kv_heads": instance.num_kv_heads, "options": dict(select_options(policy, q_len))})

    namespace = dict(vars(bc_attn)) | {"_qwasar_options": lambda q: select_options(policy, q),
                                      "_qwasar_capture_config": record_config}
    source = rewrite_attention(inspect.getsource(original_configure), "graph", capture=True)
    exec(compile(source, "<qwasar-attention-graph>", "exec"), namespace)

    def step(instance, x, cache_seqlens, block_table, *args, **kwargs):
        result = original_step(instance, x, cache_seqlens, block_table, *args, **kwargs)
        if capture is not None:
            capture(instance, instance._qwasar_queries[x.shape[:2]], cache_seqlens, block_table)
        return result

    # Fallback gets the same launch options; no graph slot is reused after exit.
    standalone_ns = dict(vars(triton_paged)) | {"_qwasar_options": lambda q: select_options(policy, q)}
    exec(compile(rewrite_attention(inspect.getsource(original_decode), "standalone"),
                 "<qwasar-attention-fallback>", "exec"), standalone_ns)
    patched_decode = standalone_ns["paged_attn_triton_decode"]

    def decode(*args, **kwargs):
        q = args[0] if args else kwargs["q"]
        options = {k: v for k, v in select_options(policy, q.shape[1]).items() if k != "block_h"}
        return patched_decode(*args, **(kwargs | options))

    try:
        bc_attn.BCAttn._configure = namespace["_configure"]
        if capture is not None: bc_attn.BCAttn.step = step
        triton_paged.paged_attn_triton_decode = decode
        yield configs
    finally:
        bc_attn.BCAttn._configure = original_configure
        bc_attn.BCAttn.step = original_step
        triton_paged.paged_attn_triton_decode = original_decode
