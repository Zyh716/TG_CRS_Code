import pynvml
pynvml.nvmlInit()
# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.used)

# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(1)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.used)

# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(2)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.used)

# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(3)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.used)