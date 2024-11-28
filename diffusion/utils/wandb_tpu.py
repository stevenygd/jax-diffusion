import os
import tensorflow
from tensorflow.python.distribute.cluster_resolver import (
    tpu_cluster_resolver,
)
from tensorflow.python.profiler import profiler_client


def setup_tpu_metrics():
    tpu_name = os.environ.get("TPU_NAME")
    compute_zone = os.environ.get("CLOUDSDK_COMPUTE_ZONE")
    core_project = os.environ.get("CLOUDSDK_CORE_PROJECT")
    service_addr = tpu_cluster_resolver.TPUClusterResolver(
        [tpu_name], zone=compute_zone, project=core_project
    ).get_master()
    service_addr = service_addr.replace("grpc://", "").replace(":8470", ":8466")
    result = profiler_client.monitor(service_addr, duration_ms=100, level=2)
    print(result)