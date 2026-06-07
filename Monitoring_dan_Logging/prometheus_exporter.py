from prometheus_client import start_http_server
from prometheus_client import Gauge

import psutil
import time

cpu_usage = Gauge(
    "cpu_usage_percent",
    "CPU Usage"
)

memory_usage = Gauge(
    "memory_usage_percent",
    "Memory Usage"
)

disk_usage = Gauge(
    "disk_usage_percent",
    "Disk Usage"
)

start_http_server(8001)

while True:

    cpu_usage.set(
        psutil.cpu_percent()
    )

    memory_usage.set(
        psutil.virtual_memory().percent
    )

    disk_usage.set(
        psutil.disk_usage('/').percent
    )

    time.sleep(5)