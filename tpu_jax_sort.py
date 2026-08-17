import os
import time
import json
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import shard_map

process_id = int(os.environ.get("JOB_COMPLETION_INDEX", 0))
num_processes = int(os.environ.get("JOB_COMPLETIONS", 1))
coordinator_address = os.environ.get("COORDINATOR_ADDRESS", "127.0.0.1:1234")

jax.distributed.initialize(
    coordinator_address=coordinator_address,
    num_processes=num_processes,
    process_id=process_id,
)

rank = jax.process_index()
devices = jax.devices()
num_devices = len(devices)

mesh = Mesh(devices, ('x',))
sharding = NamedSharding(mesh, P('x'))

# ~33.55 GB per chip * 16 chips = 536.87 GB total
ELEMENTS_PER_CHIP = 4_194_304_000 
BYTES_PER_CHIP = ELEMENTS_PER_CHIP * 8
TOTAL_BYTES = BYTES_PER_CHIP * num_devices

@jax.jit
def sort_and_exchange(data):
    local_sorted = jnp.sort(data, axis=-1)
    exchanged = jax.lax.all_to_all(local_sorted, axis_name='x', split_axis=1, concat_axis=1)
    return jnp.sort(exchanged, axis=-1)

def main():
    prng_keys = jax.random.split(jax.random.PRNGKey(42), num_devices)
    distributed_seeds = jax.device_put(prng_keys, sharding)

    @jax.jit
    def generate_keys(seeds):
        return jax.vmap(lambda s: jax.random.bits(s, shape=(ELEMENTS_PER_CHIP,), dtype=jnp.uint64))(seeds)

    tpu_keys = generate_keys(distributed_seeds)
    tpu_keys.block_until_ready()

    # Warmup / Compilation
    with mesh:
        warmup = shard_map.shard_map(sort_and_exchange, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_rep=False)(tpu_keys)
        warmup.block_until_ready()

    if rank == 0:
        jax.profiler.start_trace("/tmp/tpu_profile_logs")

    start_time = time.perf_counter()

    with mesh:
        sorted_keys = shard_map.shard_map(sort_and_exchange, mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_rep=False)(tpu_keys)
        sorted_keys.block_until_ready()

    end_time = time.perf_counter()

    if rank == 0:
        jax.profiler.stop_trace()
        total_time = end_time - start_time
        metrics = {
            "option": "Option 2 (Native TPU v6e Silicon)",
            "data_gb": TOTAL_BYTES / 1e9,
            "execution_seconds": round(total_time, 4),
            "throughput_gbps": round((TOTAL_BYTES / 1e9) / total_time, 2),
            "compliant": False,
            "memory_location": "TPU HBM3",
            "interconnect": "800 GB/s ICI Fabric"
        }
        with open("/tmp/option2_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Option 2 completed successfully in {total_time:.4f} seconds.")

if __name__ == "__main__":
    main()