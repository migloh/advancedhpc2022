from numba import cuda
print("Check if there is an available GPU:")
print(cuda.is_available())
print('\nView all GPU:')
print(cuda.detect())

cores_per_sm = 64
device_0 = cuda.select_device(0)
print('Device name: ', device_0.name)
print('Multiprocessor count: ', device_0.MULTIPROCESSOR_COUNT)
print('Core count:', device_0.MULTIPROCESSOR_COUNT*cores_per_sm)
print('Total memory size: ', cuda.current_context().get_memory_info().total, 'bytes')